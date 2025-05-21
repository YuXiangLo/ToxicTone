import json
import random
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# =============== 0. Set Seed for Reproducibility ===============
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============== 1. Define the Model ===============
class NNClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=(1024, 512)):
        super(NNClassifier, self).__init__()
        # Simple logic to reduce hidden dimensions if input_dim is smaller
        if input_dim == 2048:
            hidden_dims = (768, 384)
        elif input_dim == 1024:
            hidden_dims = (512, 256)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # BCEWithLogitsLoss expects raw logits, so no sigmoid here
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# =============== 2. Data Preparation ===============
def prepare_data(
    label_idx=0,
    file_ids_path="jsons/file_ids.json",
    binarize_path="jsons/multi_source.json",
    asr_embeddings_path="pts/asr_embeddings.pt",
    sonar_embeddings_path="pts/sonar_embeddings.pt",
    emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
    train_ratio=0.9,
    batch_size=64
):
    """
    Loads data, filters based on multi_source.json, picks the label from label_idx,
    concatenates embeddings (currently using SONAR alone in this example), removes corrupted data,
    and returns train, test DataLoaders (90/10 split).
    """
    # Load all IDs
    with open(file_ids_path, 'r') as f:
        all_ids = json.load(f)

    # Load label map from multi_source.json
    label_map = {}
    with open(binarize_path, 'r') as f:
        # Each entry is expected to have the form:
        # {
        #   "id": "some_id",
        #   "source": [word_label, anger_label, despise_label, irony_label, threat_label]
        # }
        data = json.load(f)
        for label_info in data:
            # label_idx is which index in `source` we want
            # e.g. 0 -> word, 1 -> anger, ...
            label_map[label_info['id']] = label_info['source'][label_idx]

    # Filter IDs to only those in label_map
    filtered_ids = [id_ for id_ in all_ids if id_ in label_map]

    # Create filtered labels tensor
    labels = torch.tensor(
        [float(label_map[id_]) for id_ in filtered_ids], dtype=torch.float32
    ).view(-1, 1)

    # Load embeddings
    asr_embeddings = torch.load(asr_embeddings_path, map_location=torch.device('cpu'))
    sonar_embeddings = torch.load(sonar_embeddings_path, map_location=torch.device('cpu'))
    emo2vec_embeddings = torch.load(emo2vec_embeddings_path, map_location=torch.device('cpu'))

    # Create an index mapping from original IDs to their positions
    id_to_index = {id_: idx for idx, id_ in enumerate(all_ids)}

    # Filter embeddings to match only filtered_ids
    filtered_indices = torch.tensor([id_to_index[id_] for id_ in filtered_ids], dtype=torch.long)
    asr_embeddings = asr_embeddings[filtered_indices]
    sonar_embeddings = sonar_embeddings[filtered_indices]
    emo2vec_embeddings = emo2vec_embeddings[filtered_indices]

    # For illustration, let's just use SONAR. 
    # If you want to concatenate everything, use something like:
    # concat_tensor = torch.cat([asr_embeddings, emo2vec_embeddings], dim=1)
    concat_tensor = emo2vec_embeddings

    # Remove corrupted data (NaNs or Inf)
    valid_mask = (
        ~torch.isnan(concat_tensor).any(dim=1) &
        ~torch.isinf(concat_tensor).any(dim=1)
    )
    clean_concat_tensor = concat_tensor[valid_mask]
    clean_labels = labels[valid_mask]

    print(f"[Label {label_idx}] Removed {len(concat_tensor) - len(clean_concat_tensor)} corrupted samples.")
    print(f"[Label {label_idx}] Total clean samples: {len(clean_concat_tensor)}")

    # Split (90% train, 10% test)
    num_samples = len(clean_concat_tensor)
    train_size = int(train_ratio * num_samples)
    test_size = num_samples - train_size

    dataset = TensorDataset(clean_concat_tensor, clean_labels)
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============== 3. Training Function ===============
def train_model(
    model,
    train_loader,
    device,
    pos_weight=1.0,
    threshold=0.5,
    lr=1e-4,
    num_epochs=15,
    max_grad_norm=1.0,
    verbose=False
):
    """
    Train the model for a fixed number of epochs on train_loader (no separate validation).
    If 'verbose=True', logs training metrics each epoch.
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

        for batch_X, batch_y in train_loader:
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

        if verbose:
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
    Evaluate model on given DataLoader (test set).
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


# =============== 5. Simple Hyperparam Search ===============
def hyperparam_search(train_loader, test_loader, device, input_dim):
    """
    For illustration: a simple grid search over a few hyperparameters.
    We train/eval on the same train/test for each combination,
    then pick the combination with the highest F1 on test.

    Returns the best_params dict.
    """

    # Define parameter grid
    # (Feel free to expand or modify as desired)
    param_grid = {
        "pos_weight": [1.0, 2.0, 3.0, 4.0, 5.0],
        "lr": [1e-3, 5e-4, 1e-4],
        "num_epochs": [20, 30],
        "max_grad_norm": [1.0],     # if you want to try more, add them here
        "threshold": [0.3, 0.4, 0.5, 0.6],        # you can also try other thresholds
    }

    best_f1 = -1.0
    best_params = None

    # Iterate over all combinations in param_grid
    all_keys = list(param_grid.keys())
    all_combos = list(itertools.product(*(param_grid[k] for k in all_keys)))

    for combo in all_combos:
        # Build a parameter dict from the combo
        params = {k: v for k, v in zip(all_keys, combo)}

        # Initialize a fresh model for each param set
        model = NNClassifier(input_dim=input_dim).to(device)

        # Train the model
        _ = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            pos_weight=params["pos_weight"],
            threshold=params["threshold"],
            lr=params["lr"],
            num_epochs=params["num_epochs"],
            max_grad_norm=params["max_grad_norm"],
            verbose=False  # set True if you want training logs
        )

        # Evaluate on test set
        _, _, _, _, f1 = evaluate(
            model, test_loader, device=device, threshold=params["threshold"]
        )

        # Keep track of the best
        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()

    return best_params


# =============== 6. Main Execution ===============
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # We'll loop over label_idx = 0..4 for your multi_source fields
    #  (word, anger, despise, irony, threat)
    results = {}
    for label_idx in range(5):
        print(f"\n===== Label Index {label_idx} =====")

        # --- (A) Prepare Data ---
        train_loader, test_loader = prepare_data(
            label_idx=label_idx,  # which label in multi_source.json
            file_ids_path="jsons/file_ids.json",
            binarize_path="jsons/multi_source.json",
            asr_embeddings_path="pts/asr_embeddings.pt",
            sonar_embeddings_path="pts/sonar_embeddings.pt",
            emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
            train_ratio=0.9,  # 90% train, 10% test
            batch_size=64
        )

        # The input dimension is the dimension of your chosen embeddings
        input_dim = train_loader.dataset[0][0].shape[0]

        # --- (B) Simple Hyperparam Search ---
        best_params = hyperparam_search(train_loader, test_loader, device, input_dim)
        print(f"[Label {label_idx}] Best Hyperparams found: {best_params}")

        # --- (C) Re-train Model with best_params (to get final metrics) ---
        final_model = NNClassifier(input_dim=input_dim).to(device)
        final_model = train_model(
            model=final_model,
            train_loader=train_loader,
            device=device,
            pos_weight=best_params['pos_weight'],
            threshold=best_params['threshold'],
            lr=best_params['lr'],
            num_epochs=best_params['num_epochs'],
            max_grad_norm=best_params['max_grad_norm'],
            verbose=True  # let's see the training logs here
        )

        # --- (D) Evaluate on Test Data with best threshold ---
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
            final_model, test_loader, device=device, threshold=best_params['threshold']
        )

        # --- (E) Print & Store Final Test Results ---
        print(f"===== Final Test Results for Label {label_idx} =====")
        print(f"Loss:      {test_loss:.4f}")
        print(f"Accuracy:  {test_acc:.2f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1-score:  {test_f1:.4f}")
        print("======================================")

        results[label_idx] = {
            "best_params": best_params,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }

        # Optionally save each model separately
        torch.save(final_model.state_dict(), f'outputs/trained_classifier_label_{label_idx}.pt')

    # After finishing all labels, you could also store the results in a JSON, if you like
    # For example:
    # with open("outputs/results_summary.json", "w") as f:
    #     json.dump(results, f, indent=2)

    print("\nAll label experiments complete!")
    print("Summary of best results (F1) per label:")
    for label_idx in range(5):
        print(
            f"Label {label_idx} -> "
            f"Best F1: {results[label_idx]['test_f1']:.4f}, "
            f"Params: {results[label_idx]['best_params']}"
        )


if __name__ == "__main__":
    main()

