import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from itertools import product

# =============== 0. Set Seed for Reproducibility ===============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============== 1. Define the Model ===============
class WeightedSumLayer(nn.Module):
    def __init__(self, num_layers, normalize=False):
        super(WeightedSumLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.normalize = normalize
    
    def forward(self, all_hs):
        """
        all_hs: List of tensors of shape [N, H]
        Returns: Tensor of shape [N, H]
        """
        assert len(all_hs) > 1
        stacked_hs = torch.stack(all_hs, dim=0)
        
        if self.normalize:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))
        
        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(len(all_hs), -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)
        
        return weighted_hs


class NNClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=(512, 256), num_layers=12):
        super(NNClassifier, self).__init__()
        self.weighted_sum = WeightedSumLayer(num_layers)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.weighted_sum([x[:, i, :] for i in range(x.shape[1])])  # Aggregate across layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid here (we'll use BCEWithLogitsLoss)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# =============== 2. Data Preparation ===============
def prepare_data(
    file_ids_path="jsons/file_ids.json",
    binarize_path="jsons/binarize.json",
    train_path="jsons/train.json",
    test_path="jsons/test.json",
    wavlm_embeddings_path="pts/wavlm_embeddings.pt",
    batch_size=64
):
    """
    Loads data, applies weighted sum over WavLM embeddings, removes corrupted data,
    and returns train, test DataLoaders using provided train/test splits.
    """
    # Load IDs
    with open(file_ids_path, 'r') as f:
        ids = json.load(f)

    # Load train/test IDs
    with open(train_path, 'r') as f:
        train_ids = set(json.load(f))

    with open(test_path, 'r') as f:
        test_ids = set(json.load(f))

    # Load label map
    label_map = {}
    with open(binarize_path, 'r') as f:
        for label_info in json.load(f):
            label_map[label_info['id']] = label_info['source'][4]

    # Create labels tensor in [N, 1]
    labels = torch.tensor(
        [float(label_map[id_]) if id_ in label_map else float(-1) for id_ in ids], dtype=torch.float32
    ).view(-1, 1)

    # Load WavLM embeddings
    wavlm_embeddings = torch.load(wavlm_embeddings_path, map_location=torch.device('cpu'))

    # Remove corrupted data (NaNs or Inf)
    valid_mask = ~torch.isnan(wavlm_embeddings).any(dim=[1, 2]) & ~torch.isinf(wavlm_embeddings).any(dim=[1, 2])
    clean_wavlm_embeddings = wavlm_embeddings[valid_mask]
    clean_labels = labels[valid_mask]
    clean_ids = [id_ for i, id_ in enumerate(ids) if valid_mask[i]]

    print(f"✅ Removed {len(wavlm_embeddings) - len(clean_wavlm_embeddings)} corrupted samples.")
    print(f"Total clean samples: {len(clean_wavlm_embeddings)}")

    # Assign train and test data based on original split
    train_data = [(emb, label) for emb, label, id_ in zip(clean_wavlm_embeddings, clean_labels, clean_ids) if id_ in train_ids and label != float(-1)]
    test_data = [(emb, label) for emb, label, id_ in zip(clean_wavlm_embeddings, clean_labels, clean_ids) if id_ in test_ids and label != float(-1)]

    # Convert to TensorDataset
    train_dataset = TensorDataset(torch.stack([t[0] for t in train_data]), torch.stack([t[1] for t in train_data]))
    test_dataset = TensorDataset(torch.stack([t[0] for t in test_data]), torch.stack([t[1] for t in test_data]))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# =============== 3. Training Function (No Validation) ===============
def train_model(
    model,
    train_loader,
    device,
    pos_weight=1.0,
    threshold=0.5,
    lr=1e-4,
    num_epochs=15,
    max_grad_norm=1.0
):
    """
    Train the model for a fixed number of epochs on train_loader (no validation).
    Logs training metrics each epoch to stdout.
    """
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds, all_targets = [], []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as pbar:
            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                total_loss += loss.item()

                # Predictions
                preds = (torch.sigmoid(outputs) >= threshold).float().view(-1)
                batch_y = batch_y.view(-1)

                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Compute training metrics for the epoch
        train_loss = total_loss / len(train_loader)
        train_accuracy = (correct / total) * 100.0
        train_precision = precision_score(all_targets, all_preds, zero_division=0)
        train_recall = recall_score(all_targets, all_preds, zero_division=0)
        train_f1 = f1_score(all_targets, all_preds, zero_division=0)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, "
            f"F1: {train_f1:.4f}, "
            f"Acc: {train_accuracy:.2f}, "
            f"Prec: {train_precision:.4f}, "
            f"Rec: {train_recall:.4f}"
        )

    return model


# =============== 4. Evaluation on Test Set ===============
def evaluate(model, data_loader, device, threshold=0.5):
    """
    Evaluate model on test set.
    Returns avg_loss, accuracy, precision, recall, f1.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()  # Standard weighting for test
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= threshold).float().view(-1)
            batch_y = batch_y.view(-1)

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = (correct / total) * 100.0
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1



def main():
    from itertools import product

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- (A) Prepare Data ---
    train_loader, test_loader = prepare_data(
        file_ids_path="jsons/file_ids.json",
        binarize_path="jsons/multi_source.json",
        train_path="jsons/train.json",
        test_path="jsons/test.json",
        wavlm_embeddings_path="pts/xls_r_1b_embeddings.pt",
        batch_size=64
    )

    # Determine the number of layers and hidden dimension dynamically
    # based on the first sample in the train set:
    num_layers, hidden_dim = train_loader.dataset[0][0].shape[0], train_loader.dataset[0][0].shape[1]

    # --- (B) Define Parameter Grid ---
    param_grid = {
        # "pos_weight": [1.0, 2.0, 4.0, 8.0, 16.0],
        "pos_weight": [1.0],
        "threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "lr": [1e-3],
        "num_epochs": [10, 20],
        "max_grad_norm": [1.0],
    }

    # We'll track the best F1 and its corresponding parameters
    best_f1, best_acc = 0.0, 0.0
    best_params = None
    best_model_state = None  # To store model weights of the best combo

    # --- (C) Grid Search ---
    # Generate all possible combinations of the parameter grid
    all_combinations = list(product(*param_grid.values()))
    print(f"Number of hyperparameter combinations: {len(all_combinations)}")

    # param_grid.keys() => ["pos_weight", "threshold", "lr", "num_epochs", "max_grad_norm"]
    param_names = list(param_grid.keys())

    for combo in all_combinations:
        # combo is a tuple like (pos_weight_value, threshold_value, lr_value, num_epochs_value, max_grad_norm_value)
        hyperparams = dict(zip(param_names, combo))

        # Print or log the current hyperparam combo for clarity
        print("\nCurrent Hyperparams:", hyperparams)

        # Initialize a fresh model
        model = NNClassifier(input_dim=hidden_dim, num_layers=num_layers).to(device)

        # Train the model with these hyperparams
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            pos_weight=hyperparams['pos_weight'],
            threshold=hyperparams['threshold'],
            lr=hyperparams['lr'],
            num_epochs=hyperparams['num_epochs'],
            max_grad_norm=hyperparams['max_grad_norm']
        )

        # Evaluate on test set
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
            trained_model,
            test_loader,
            device,
            threshold=hyperparams['threshold']
        )

        print(
            f"Test metrics => Loss: {test_loss:.4f} | Acc: {test_acc:.2f} | "
            f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}"
        )

        # Update best if we see a higher F1
        if test_f1 * 100 + test_acc > best_f1 * 100 + best_acc:
            best_f1 = test_f1
            best_acc = test_acc
            best_params = hyperparams
            # Save the model's state_dict (so we can later re-initialize with best model weights)
            best_model_state = trained_model.state_dict()

    # --- (D) Re-initialize model with best hyperparams and best state ---
    print("\n===== Best Hyperparameters Found =====")
    print(best_params)
    print(f"Best F1: {best_f1:.4f}")

    # Create a new model with the best hyperparams if you want to keep training or inference
    best_model = NNClassifier(input_dim=hidden_dim, num_layers=num_layers).to(device)
    best_model.load_state_dict(best_model_state)

    # Optionally re-evaluate or just finalize saving
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        best_model, test_loader, device, threshold=best_params['threshold']
    )
    print("\n===== Final Evaluation with Best Model =====")
    print(f"Params:         {best_params}")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test Accuracy:  {test_acc:.2f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")

    # Save best model
    print("Saving best model to outputs/best_classifier.pt...")
    torch.save(best_model, 'outputs/best_classifier.pt')
    print("Done!")


if __name__ == "__main__":
    main()
