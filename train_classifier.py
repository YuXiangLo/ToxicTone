import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

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
        x = self.fc3(x)  # No sigmoid here (we'll use BCEWithLogitsLoss)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# =============== 2. Data Preparation (Train/Test = 90/10) ===============
def prepare_data(
    file_ids_path="jsons/file_ids.json",
    binarize_path="jsons/binarize.json",
    train_path="jsons/train.json",
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

    with open(test_path, 'r') as f:
        test_ids = set(json.load(f))

    # Load label map
    label_map = {}
    with open(binarize_path, 'r') as f:
        for label_info in json.load(f):
            label_map[label_info['id']] = label_info['toxicity']

    # Create labels tensor in [N, 1]
    labels = torch.tensor(
        [float(label_map[id_]) for id_ in ids], dtype=torch.float32
    ).view(-1, 1)

    # Load embeddings
    asr_embeddings = torch.load(asr_embeddings_path, map_location=torch.device('cpu'))
    sonar_embeddings = torch.load(sonar_embeddings_path, map_location=torch.device('cpu'))
    emo2vec_embeddings = torch.load(emo2vec_embeddings_path, map_location=torch.device('cpu'))

    # Concatenate embeddings
    # concat_tensor = torch.cat([sonar_embeddings, emo2vec_embeddings], dim=1)
    concat_tensor = emo2vec_embeddings

    # Remove corrupted data (NaNs or Inf)
    valid_mask = ~torch.isnan(concat_tensor).any(dim=1) & ~torch.isinf(concat_tensor).any(dim=1)
    clean_concat_tensor = concat_tensor[valid_mask]
    clean_labels = labels[valid_mask]
    clean_ids = [id_ for i, id_ in enumerate(ids) if valid_mask[i]]

    print(f"✅ Removed {len(concat_tensor) - len(clean_concat_tensor)} corrupted samples.")
    print(f"Total clean samples: {len(clean_concat_tensor)}")

    # Assign train and test data based on predefined split
    train_data = [(emb, label) for emb, label, id_ in zip(clean_concat_tensor, clean_labels, clean_ids) if id_ in train_ids]
    test_data = [(emb, label) for emb, label, id_ in zip(clean_concat_tensor, clean_labels, clean_ids) if id_ in test_ids]

    # Convert to TensorDataset
    train_dataset = TensorDataset(torch.stack([t[0] for t in train_data]), torch.stack([t[1] for t in train_data]))
    test_dataset = TensorDataset(torch.stack([t[0] for t in test_data]), torch.stack([t[1] for t in test_data]))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# def prepare_data(
#     file_ids_path="jsons/file_ids.json",
#     binarize_path="jsons/binarize.json",
#     asr_embeddings_path="pts/asr_embeddings.pt",
#     sonar_embeddings_path="pts/sonar_embeddings.pt",
#     emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
#     train_ratio=0.9,
#     batch_size=64
# ):
#     """
#     Loads data, concatenates embeddings, removes corrupted data,
#     and returns train, test DataLoaders (90/10 split).
#     """
#     # Load IDs
#     with open(file_ids_path, 'r') as f:
#         ids = json.load(f)

#     # Load label map
#     label_map = {}
#     with open(binarize_path, 'r') as f:
#         for label_info in json.load(f):
#             label_map[label_info['id']] = label_info['toxicity']

#     # Create labels tensor in [N, 1]
#     labels = torch.tensor(
#         [float(label_map[id_]) for id_ in ids], dtype=torch.float32
#     ).view(-1, 1)

#     # Load embeddings and concatenate
#     asr_embeddings = torch.load(asr_embeddings_path, map_location=torch.device('cpu'))
#     sonar_embeddings = torch.load(sonar_embeddings_path, map_location=torch.device('cpu'))
#     emo2vec_embeddings = torch.load(emo2vec_embeddings_path, map_location=torch.device('cpu'))
#     # concat_tensor = torch.cat([sonar_embeddings, emo2vec_embeddings], dim=1)
#     concat_tensor = sonar_embeddings

#     # Remove corrupted data (NaNs or Inf)
#     valid_mask = ~torch.isnan(concat_tensor).any(dim=1) & ~torch.isinf(concat_tensor).any(dim=1)
#     clean_concat_tensor = concat_tensor[valid_mask]
#     clean_labels = labels[valid_mask]

#     print(f"✅ Removed {len(concat_tensor) - len(clean_concat_tensor)} corrupted samples.")
#     print(f"Total clean samples: {len(clean_concat_tensor)}")

#     # Split (90% train, 10% test)
#     num_samples = len(clean_concat_tensor)
#     train_size = int(train_ratio * num_samples)
#     test_size = num_samples - train_size

#     dataset = TensorDataset(clean_concat_tensor, clean_labels)
#     train_data, test_data = random_split(dataset, [train_size, test_size])

#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


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


# =============== 5. Main Execution ===============
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- (A) Prepare Data (only Train & Test) ---
    train_loader, test_loader = prepare_data(
        file_ids_path="jsons/file_ids.json",
        binarize_path="jsons/binarize.json",
        asr_embeddings_path="pts/asr_embeddings.pt",
        sonar_embeddings_path="pts/sonar_embeddings.pt",
        emo2vec_embeddings_path="pts/emo2vec_embeddings.pt",
        # train_ratio=0.9,  # 90% train, 10% test
        batch_size=64
    )

    # --- (B) Specify best param set (EDIT as needed) ---
    best_params = {
        "pos_weight": 1.0,
        "threshold": 0.4,
        "lr": 1e-3,
        "num_epochs": 20,
        "max_grad_norm": 1.0
    }

    # --- (C) Initialize Model ---
    input_dim = train_loader.dataset[0][0].shape[0]  # dimension of concatenated embeddings
    model = NNClassifier(input_dim=input_dim).to(device)

    # --- (D) Train Model with best_params ---
    model = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        pos_weight=best_params['pos_weight'],
        threshold=best_params['threshold'],
        lr=best_params['lr'],
        num_epochs=best_params['num_epochs'],
        max_grad_norm=best_params['max_grad_norm']
    )

    # --- (E) Evaluate on Test Data ---
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, device, threshold=best_params['threshold']
    )

    # --- (F) Print Final Test Results ---
    print("\n===== Final Test Results =====")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {test_acc:.2f}")
    print(f"F1-score:  {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print("==============================")

    print("Trying to save model to outputs...")
    torch.save(model, 'outputs/trained_classifier.pt')
    print("Done!")

if __name__ == "__main__":
    main()

