import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import itertools

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
        all_hs: list of tensors of shape [N, H]
        Returns: A single tensor of shape [N, H]
        """
        assert len(all_hs) > 1
        stacked_hs = torch.stack(all_hs, dim=0)  # [num_layers, N, H]

        if self.normalize:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        # Flatten and apply softmax to combine
        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(len(all_hs), -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs


class NNClassifier(nn.Module):
    def __init__(
        self,
        wavlm_dim=768,    # dim of each layer's embedding
        sonar_dim=128,    # dim of sonar embedding
        emo2vec_dim=64,   # dim of emo2vec embedding
        hidden_dims=(768, 384),
        num_layers=12
    ):
        super(NNClassifier, self).__init__()

        # Weighted sum for the WavLM embeddings (num_layers of them)
        self.weighted_sum = WeightedSumLayer(num_layers)

        # Total dimension going into the first FC layer
        total_input_dim = wavlm_dim + sonar_dim + emo2vec_dim
        self.fc1 = nn.Linear(total_input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, wavlm, sonar, emo2vec):
        """
        wavlm:   Tensor [N, num_layers, wavlm_dim]
        sonar:   Tensor [N, sonar_dim]
        emo2vec: Tensor [N, emo2vec_dim]
        Returns: (Logits) Tensor [N, 1]
        """
        # Weighted-sum across the multiple WavLM layers
        x_agg = self.weighted_sum([wavlm[:, i, :] for i in range(wavlm.shape[1])])

        # Concatenate with the other embeddings
        combined = torch.cat((x_agg, sonar, emo2vec), dim=1)

        # Pass through MLP
        out = self.relu(self.fc1(combined))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)  # BCEWithLogitsLoss expects raw logits
        return out

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
    sonar_embeddings_path="pts/sonar_embeddings.pt",
    emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
    batch_size=64
):
    """
    Loads data, applies filtering (removing NaN/Inf), splits by train/test,
    and returns train, test DataLoaders with:
      - (wavlm_emb, sonar_emb, emo2vec_emb, label)
    """

    # Load IDs
    with open(file_ids_path, 'r') as f:
        ids = json.load(f)

    # Load train/test splits
    with open(train_path, 'r') as f:
        train_ids = set(json.load(f))

    with open(test_path, 'r') as f:
        test_ids = set(json.load(f))

    # Load label map
    label_map = {}
    with open(binarize_path, 'r') as f:
        for label_info in json.load(f):
            label_map[label_info['id']] = label_info['source'][4]

    # Create labels tensor
    labels = torch.tensor(
        [float(label_map[id_]) if id_ in label_map else float(-1) for id_ in ids], dtype=torch.float32
    ).view(-1, 1)

    # Load embeddings
    wavlm_embeddings = torch.load(wavlm_embeddings_path, map_location=torch.device('cpu'))
    sonar_embeddings = torch.load(sonar_embeddings_path, map_location=torch.device('cpu'))
    emo2vec_embeddings = torch.load(emo2vec_embeddings_path, map_location=torch.device('cpu'))

    # Ensure all have same length
    assert len(wavlm_embeddings) == len(sonar_embeddings) == len(emo2vec_embeddings) == len(labels), \
        "Mismatch in number of samples between embeddings or labels."

    # Filter out corrupted data (NaN or Inf in any embedding)
    valid_mask = (
        ~torch.isnan(wavlm_embeddings).any(dim=[1, 2]) &
        ~torch.isinf(wavlm_embeddings).any(dim=[1, 2]) &
        ~torch.isnan(sonar_embeddings).any(dim=1) &
        ~torch.isinf(sonar_embeddings).any(dim=1) &
        ~torch.isnan(emo2vec_embeddings).any(dim=1) &
        ~torch.isinf(emo2vec_embeddings).any(dim=1)
    )

    clean_wavlm_embeddings = wavlm_embeddings[valid_mask]
    clean_sonar_embeddings = sonar_embeddings[valid_mask]
    clean_emo2vec_embeddings = emo2vec_embeddings[valid_mask]
    clean_labels = labels[valid_mask]
    clean_ids = [id_ for i, id_ in enumerate(ids) if valid_mask[i]]

    print(f"✅ Removed {len(wavlm_embeddings) - len(clean_wavlm_embeddings)} corrupted samples.")
    print(f"Total clean samples: {len(clean_wavlm_embeddings)}")

    # Split into train and test based on original ID-based split
    train_data = []
    test_data = []

    for emb_wavlm, emb_sonar, emb_emo, label, _id in zip(
        clean_wavlm_embeddings, clean_sonar_embeddings, clean_emo2vec_embeddings, clean_labels, clean_ids
    ):
        if label == float(-1): continue
        if _id in train_ids:
            train_data.append((emb_wavlm, emb_sonar, emb_emo, label))
        elif _id in test_ids:
            test_data.append((emb_wavlm, emb_sonar, emb_emo, label))

    # Convert to Tensors
    train_wavlm = torch.stack([t[0] for t in train_data])
    train_sonar = torch.stack([t[1] for t in train_data])
    train_emo2vec = torch.stack([t[2] for t in train_data])
    train_labels = torch.stack([t[3] for t in train_data])

    test_wavlm = torch.stack([t[0] for t in test_data])
    test_sonar = torch.stack([t[1] for t in test_data])
    test_emo2vec = torch.stack([t[2] for t in test_data])
    test_labels = torch.stack([t[3] for t in test_data])

    # Build Datasets
    train_dataset = TensorDataset(train_wavlm, train_sonar, train_emo2vec, train_labels)
    test_dataset = TensorDataset(test_wavlm, test_sonar, test_emo2vec, test_labels)

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
    Logs training metrics each epoch.
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
            for batch in pbar:
                # Unpack the batch
                batch_wavlm, batch_sonar, batch_emo, batch_y = (t.to(device) for t in batch)

                optimizer.zero_grad()
                outputs = model(batch_wavlm, batch_sonar, batch_emo)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                total_loss += loss.item()

                # Predictions
                preds = (torch.sigmoid(outputs) >= threshold).float().view(-1)
                batch_y_flat = batch_y.view(-1)

                correct += (preds == batch_y_flat).sum().item()
                total += batch_y_flat.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y_flat.cpu().numpy())

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
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch_wavlm, batch_sonar, batch_emo, batch_y = (t.to(device) for t in batch)
            outputs = model(batch_wavlm, batch_sonar, batch_emo)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= threshold).float().view(-1)
            batch_y_flat = batch_y.view(-1)

            correct += (preds == batch_y_flat).sum().item()
            total += batch_y_flat.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y_flat.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = (correct / total) * 100.0
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


# =============== 5. Main Script ===============

def grid_search(param_grid, device, train_loader, test_loader):
    """
    param_grid: dict of hyperparameters where each value is a list of possible values
                e.g. {
                  'lr': [1e-3, 1e-4],
                  'pos_weight': [1.0, 2.0],
                  'threshold': [0.4, 0.5],
                  'num_epochs': [10, 20],
                  'max_grad_norm': [1.0, None]
                }
    device:  torch device
    train_loader, test_loader: DataLoader objects
    """
    # We will store results as a list of (param_set, metrics_dict)
    results = []
    # Build up all combinations of parameters
    keys = list(param_grid.keys())
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        # Create a dictionary of this particular parameter combo
        params = dict(zip(keys, combo))
        print("\n============================================")
        print("Trying param set:", params)
        print("============================================")

        # 1) Initialize a new model for each combination
        #    We should use the same shapes as in your main().
        #    For demonstration, we'll do a quick pass with the shape from the train_loader.
        example_wavlm, example_sonar, example_emo, _ = next(iter(train_loader))
        num_layers, wavlm_dim = example_wavlm.shape[1], example_wavlm.shape[2]
        sonar_dim = example_sonar.shape[1]
        emo2vec_dim = example_emo.shape[1]

        model = NNClassifier(
            wavlm_dim=wavlm_dim,
            sonar_dim=sonar_dim,
            emo2vec_dim=emo2vec_dim,
            hidden_dims=(512, 256),   # or any structure you like
            num_layers=num_layers
        ).to(device)

        # 2) Train with the given hyperparameters
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            pos_weight=params["pos_weight"],
            threshold=params["threshold"],
            lr=params["lr"],
            num_epochs=params["num_epochs"],
            max_grad_norm=params["max_grad_norm"]
        )

        # 3) Evaluate the trained model on the test set
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            trained_model,
            data_loader=test_loader,
            device=device,
            threshold=params["threshold"]
        )

        # 4) Store the results
        metrics_dict = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "test_f1": test_f1
        }
        results.append((params, metrics_dict))
        print(
            f"--> Results: F1={test_f1:.4f}, Acc={test_acc:.2f}, "
            f"Prec={test_prec:.4f}, Rec={test_rec:.4f}, Loss={test_loss:.4f}"
        )

    # 5) Find the best performing parameter set (by F1, for instance)
    best_params, best_metrics = max(results, key=lambda x: x[1]["test_f1"] * 100 + x[1]["test_acc"])
    print("\n********** GRID SEARCH COMPLETE **********")
    print(f"Best Param Set (by F1): {best_params}")
    print(f"Best Test F1:          {best_metrics['test_f1']:.4f}")
    print("All metrics for best set:", best_metrics)

    return best_params, best_metrics, results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- (A) Prepare Data ---
    train_loader, test_loader = prepare_data(
        file_ids_path="jsons/file_ids.json",
        binarize_path="jsons/multi_source.json",
        train_path="jsons/train.json",
        test_path="jsons/test.json",
        wavlm_embeddings_path="pts/xls_r_1b_embeddings.pt",
        sonar_embeddings_path="pts/asr_embeddings.pt",
        emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
        batch_size=64
    )
    
    # Define your parameter search space
    param_grid = {
        "lr": [1e-3],
        "pos_weight": [1.0],
        "threshold": [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "num_epochs": [10, 20],
        "max_grad_norm": [1.0]
    }

    # Run the grid search
    best_params, best_metrics, all_results = grid_search(
        param_grid=param_grid,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader
    )

    # best_params is a dictionary with the hyperparameters that gave the best F1
    # best_metrics is the metric dictionary for that best run
    # all_results is a list of (params, metrics) for all tested combos

    # If you want to do a final train with best_params:
    # (Re-initialize a new model, train, then save it.)
    print("\nRe-training on best hyperparams...")

    example_wavlm, example_sonar, example_emo, _ = next(iter(train_loader))
    num_layers, wavlm_dim = example_wavlm.shape[1], example_wavlm.shape[2]
    sonar_dim = example_sonar.shape[1]
    emo2vec_dim = example_emo.shape[1]

    final_model = NNClassifier(
        wavlm_dim=wavlm_dim,
        sonar_dim=sonar_dim,
        emo2vec_dim=emo2vec_dim,
        hidden_dims=(512, 256),
        num_layers=num_layers
    ).to(device)

    final_model = train_model(
        model=final_model,
        train_loader=train_loader,
        device=device,
        pos_weight=best_params['pos_weight'],
        threshold=best_params['threshold'],
        lr=best_params['lr'],
        num_epochs=best_params['num_epochs'],
        max_grad_norm=best_params['max_grad_norm']
    )

    # Evaluate final model on test set
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        final_model,
        test_loader,
        device,
        threshold=best_params['threshold']
    )

    print("\n===== Final Model with Best Hyperparams =====")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test Accuracy:  {test_acc:.2f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print("=============================================")

    # Save if desired
    torch.save(final_model.state_dict(), "outputs/best_model.pt")
    print("Best model saved!")


if __name__ == "__main__":
    main()

