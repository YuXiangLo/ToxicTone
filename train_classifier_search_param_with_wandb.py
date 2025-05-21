import json
import random
import itertools
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score, precision_score, recall_score

# =============== New: wandb import ===============
import wandb

# =============== 0. Set Seed for Reproducibility ===============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============== 1. Define the Model ===============
class NNClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=(1024, 512)):
        super(NNClassifier, self).__init__()
        if input_dim == 2048:
            hidden_dims = (768, 384)
        if input_dim == 1024:
            hidden_dims = (512, 256)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid here (Using BCEWithLogitsLoss)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

# =============== 2. Data Loading & Preparation ===============
def prepare_data(
    label=0,
    file_ids_path="jsons/file_ids.json",
    binarize_path="jsons/binarize.json",
    train_path="jsons/train.json",
    val_path="jsons/dev.json",
    test_path="jsons/test.json",
    asr_embeddings_path="pts/asr_embeddings.pt",
    sonar_embeddings_path="pts/sonar_embeddings.pt",
    emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
    batch_size=64
):
    """
    Loads data, concatenates embeddings, removes corrupted data,
    and returns train, test DataLoaders based on predefined splits.
    """
    # Load IDs
    with open(file_ids_path, 'r') as f:
        ids = json.load(f)

    # Load train/test IDs
    with open(train_path, 'r') as f:
        train_ids = set(json.load(f))

    # Load train/test IDs
    with open(val_path, 'r') as f:
        val_ids = set(json.load(f))

    with open(test_path, 'r') as f:
        test_ids = set(json.load(f))

    # Load label map
    label_map = {}
    with open(binarize_path, 'r') as f:
        for label_info in json.load(f):
            label_map[label_info['id']] = label_info['source'][label]

    # Create labels tensor in [N, 1]
    labels = torch.tensor(
        [float(label_map[id_]) if id_ in label_map else float(-1) for id_ in ids], dtype=torch.float32
    ).view(-1, 1)

    # Load embeddings
    asr_embeddings = torch.load(asr_embeddings_path, map_location=torch.device('cpu'))
    sonar_embeddings = torch.load(sonar_embeddings_path, map_location=torch.device('cpu'))
    emo2vec_embeddings = torch.load(emo2vec_embeddings_path, map_location=torch.device('cpu'))

    # Concatenate embeddings
    concat_tensor = torch.cat([asr_embeddings, emo2vec_embeddings], dim=1)
    # concat_tensor = asr_embeddings

    # Remove corrupted data (NaNs or Inf)
    valid_mask = ~torch.isnan(concat_tensor).any(dim=1) & ~torch.isinf(concat_tensor).any(dim=1)
    clean_concat_tensor = concat_tensor[valid_mask]
    clean_labels = labels[valid_mask]
    clean_ids = [id_ for i, id_ in enumerate(ids) if valid_mask[i]]

    print(f"✅ Removed {len(concat_tensor) - len(clean_concat_tensor)} corrupted samples.")
    print(f"Total clean samples: {len(clean_concat_tensor)}")

    # Assign train and test data based on predefined split
    train_data = [(emb, label) for emb, label, id_ in zip(clean_concat_tensor, clean_labels, clean_ids) if id_ in train_ids and label != float(-1)]
    val_data = [(emb, label) for emb, label, id_ in zip(clean_concat_tensor, clean_labels, clean_ids) if id_ in val_ids and label != float(-1)]
    test_data = [(emb, label) for emb, label, id_ in zip(clean_concat_tensor, clean_labels, clean_ids) if id_ in test_ids and label != float(-1)]

    # Convert to TensorDataset
    train_dataset = TensorDataset(torch.stack([t[0] for t in train_data]), torch.stack([t[1] for t in train_data]))
    val_dataset = TensorDataset(torch.stack([t[0] for t in val_data]), torch.stack([t[1] for t in val_data]))
    test_dataset = TensorDataset(torch.stack([t[0] for t in test_data]), torch.stack([t[1] for t in test_data]))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# =============== 3. Evaluation Function ===============
def evaluate(model, data_loader, criterion, device, threshold=0.5):
    """
    Evaluate model on a given data loader.
    Returns avg_loss, accuracy, precision, recall, f1.
    """
    model.eval()
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

            # Convert logits to binary predictions using threshold
            preds = (torch.sigmoid(outputs) >= threshold).float().view(-1)
            batch_y = batch_y.view(-1)

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = (correct / total) * 100.0  # in %
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1

# =============== 4. Train & Evaluate Pipeline (with wandb logging) ===============
def train_and_evaluate(
    train_loader,
    val_loader,
    test_loader,
    device,
    pos_weight=1.0,
    threshold=0.5,
    lr=1e-4,
    num_epochs=20,
    patience=5,
    max_grad_norm=1.0
):
    """
    Train the model with given hyperparams, evaluate on val set (with early stopping),
    then finally evaluate on test set with the best val model.
    Returns a dict of final metrics on val and test sets + best model state_dict.
    """

    # Initialize model, criterion, optimizer
    input_dim = train_loader.dataset[0][0].shape[0]
    model = NNClassifier(input_dim=input_dim).to(device)

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    no_improve_count = 0
    best_model_state = None

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

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                total_loss += loss.item()

                preds = (torch.sigmoid(outputs) >= threshold).float().view(-1)
                batch_y = batch_y.view(-1)

                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Compute training metrics
        train_loss = total_loss / len(train_loader)
        train_accuracy = (correct / total) * 100.0
        train_precision = precision_score(all_targets, all_preds, zero_division=0)
        train_recall = recall_score(all_targets, all_preds, zero_division=0)
        train_f1 = f1_score(all_targets, all_preds, zero_division=0)

        # Validation metrics
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion, device, threshold=threshold
        )

        # =============== wandb log for each epoch ===============
        wandb.log({
            "Train/Loss": train_loss,
            "Train/F1": train_f1,
            "Train/Accuracy": train_accuracy,
            "Train/Precision": train_precision,
            "Train/Recall": train_recall,

            "Val/Loss": val_loss,
            "Val/F1": val_f1,
            "Val/Accuracy": val_acc,
            "Val/Precision": val_precision,
            "Val/Recall": val_recall,

            "epoch": epoch + 1
        })

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_accuracy:.2f}, "
            f"Prec: {train_precision:.4f}, Rec: {train_recall:.4f} || "
            f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.2f}, "
            f"Prec: {val_precision:.4f}, Rec: {val_recall:.4f}"
        )

        # Early stopping (based on val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_count = 0
            best_model_state = model.state_dict()
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping triggered!")
                break

    # =============== Evaluate Best Model on Test Set ===============
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val F1 = {best_val_f1:.4f}")

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, criterion, device, threshold=threshold
    )

    print(
      f"Test Results -> Loss: {test_loss:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.2f}, "
      f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}"
    )

    # =============== wandb log for test ===============
    wandb.log({
        "Test/Loss": test_loss,
        "Test/F1": test_f1,
        "Test/Accuracy": test_acc,
        "Test/Precision": test_precision,
        "Test/Recall": test_recall
    })

    metrics = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    }
    return model, metrics

# =============== 5. Hyperparameter Search (with separate wandb runs) ===============
def hyperparameter_search(param_grid, device, train_loader, val_loader, test_loader, project_name):
    """
    Given a dict of lists of possible hyperparameters, run grid search
    and return the best model/metrics based on the weighted linear combination:
        0.5 * accuracy + 0.25 * precision + 0.25 * recall
    (on the validation set).
    """
    best_overall_score = -1.0
    best_metrics = None
    best_params = None

    # Generate all combinations of hyperparameters from the grid
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))

        print("\n==========================================")
        print(f"Training with params: {params}")
        print("==========================================")

        # =============== Start a new wandb run for each parameter set ===============
        wandb_run = wandb.init(
            project=project_name,
            config=params,
            reinit=True  # Allows restarting a new run for each iteration
        )

        _, metrics = train_and_evaluate(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            pos_weight=params['pos_weight'],
            threshold=params['threshold'],
            lr=params['lr'],
            num_epochs=params['num_epochs'],
            patience=params['patience'],
            max_grad_norm=params['max_grad_norm']
        )

        # Weighted sum on Validation set
        val_acc = metrics['val_acc']
        val_precision = metrics['val_precision']
        val_recall = metrics['val_recall']
        weighted_score = 0.5 * (val_acc / 100.0) + 0.25 * val_precision + 0.25 * val_recall

        print(f"Validation Weighted Score = {weighted_score:.4f}")

        # Log Weighted Score to wandb
        wandb.log({"Val/WeightedScore": weighted_score})

        # Close out this wandb run
        wandb.finish()

        # Track best based on this new weighted score
        if weighted_score > best_overall_score:
            best_overall_score = weighted_score
            best_metrics = metrics
            best_params = params

    return best_params, best_metrics, best_overall_score

# =============== 6. Main Execution ===============
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for l in range(5):
        # --- 1) Prepare Data ---
        train_loader, val_loader, test_loader = prepare_data(
            label=l,
            file_ids_path="jsons/file_ids.json",
            binarize_path="jsons/multi_source.json",
            sonar_embeddings_path="pts/sonar_embeddings.pt",
            emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
            batch_size=64
        )

        # --- 2) Define Hyperparameter Search Space ---
        param_grid = {
            "pos_weight": [1.0, 2.0, 4.0, 8.0, 16.0],
            "threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "lr": [1e-3],
            "num_epochs": [10, 20],
            "patience": [100],
            "max_grad_norm": [1.0]
        }

        # --- 3) Run Hyperparameter Search (Weighted Linear Combination) ---
        best_params, best_metrics, best_score = hyperparameter_search(
            param_grid=param_grid,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            project_name=f"all_multi_label_{l}"
        )

        print("\n**************** Best Result (Weighted Score) ****************")
        print(f"Best Params:    {best_params}")
        print(f"Best Val Score: {best_score:.4f}")
        print("----- Validation Metrics -----")
        print(f"Val F1:         {best_metrics['val_f1']:.4f}")
        print(f"Val Acc:        {best_metrics['val_acc']:.2f}")
        print(f"Val Precision:  {best_metrics['val_precision']:.4f}")
        print(f"Val Recall:     {best_metrics['val_recall']:.4f}")

        print("----- Test Metrics -----")
        print(f"Test F1:        {best_metrics['test_f1']:.4f}")
        print(f"Test Acc:       {best_metrics['test_acc']:.2f}")
        print(f"Test Precision: {best_metrics['test_precision']:.4f}")
        print(f"Test Recall:    {best_metrics['test_recall']:.4f}")
        print("***************************************************************")

if __name__ == "__main__":
    main()

