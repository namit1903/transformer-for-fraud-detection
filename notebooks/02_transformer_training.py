# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Load processed sequence data
import pandas as pd
import os

project_root = "/content/drive/MyDrive/FraudBehaviorEmbeddings"
input_path = os.path.join(project_root, "data/processed/transformer_input.pkl")
df = pd.read_pickle(input_path)#deserialize pickel

# Convert to list format
sequences = list(df['padded_sequence'].values)
labels = list(df['label_encoded'].values)
# print(sequences)
print(labels)

# !pip install transformers -q

#STEP 4  Prepare datasets
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FraudDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': self.sequences[idx],
            'attention_mask': (self.sequences[idx] != 0).long(),  # padding mask
            'labels': self.labels[idx]
        }

X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, stratify=labels)

train_data = FraudDataset(X_train, y_train)
val_data = FraudDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

#  STEP 5: Load DistilBERT Model
from transformers import DistilBertForSequenceClassification, DistilBertConfig

num_labels = df['label_encoded'].nunique()
# print(num_labels)

config = DistilBertConfig(
 vocab_size=30522,    # Number of tokens DistilBERT understands
    n_heads=8,           # Number of attention heads (parallel attention mechanisms)
    dim=512,             # Hidden size of each token representation (default for DistilBERT)
    hidden_dim=2048,     # Size of feed-forward layers inside the Transformer block
    n_layers=6,          # Number of Transformer layers
    # num_labels=num_labels  # Number of output classes for classification
 num_labels=3# this provided me error in the training phase

)

model = DistilBertForSequenceClassification(config)


# STEP 6: Train the model

from torch.optim import AdamW
from tqdm import tqdm

# from transformers import AdamW
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f" Epoch {epoch+1} - Loss: {total_loss / len(train_loader)}")

# STEP 7: Save model
model_path = os.path.join(project_root, "models/transformer/distilbert_fraud.pt")
torch.save(model.state_dict(), model_path)
print(f" Saved model to {model_path}")
