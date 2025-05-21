import torch
import torch.nn as nn
import json

class NNClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=(768, 384)):
        super(NNClassifier, self).__init__()
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.load("outputs/trained_classifier.pt", map_location=device)

# Ensure model is in evaluation mode
model.eval()

# Load embeddings
sonar_embeddings = torch.load("pts/sonar_embeddings.pt", map_location=device)
emo2vec_embeddings = torch.load("pts/emo2vec_embeddings.pt", map_location=device)

# Concatenate embeddings
# input_tensor = torch.cat([sonar_embeddings, emo2vec_embeddings], dim=1).to(device)
input_tensor = sonar_embeddings

with torch.no_grad():
    outputs = model(input_tensor)
    predictions = torch.sigmoid(outputs)  # Convert logits to probabilities
    binary_preds = (predictions >= 0.5).float()  # Thresholding at 0.5

print("Raw Logits:", outputs.cpu().numpy())
print("Probabilities:", predictions.cpu().numpy())
print("Binary Predictions:", binary_preds.cpu().numpy())  # 1 = Toxic, 0 = Non-Toxic

