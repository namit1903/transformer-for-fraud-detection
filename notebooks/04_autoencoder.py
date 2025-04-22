from google.colab import drive
drive.mount('/content/drive')


# 📌 STEP 1: Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

root = "/content/drive/MyDrive/FraudBehaviorEmbeddings"
embeddings = np.load(os.path.join(root, "data/processed/graphsage_embeddings.npy"))
embeddings.shape

# import os
# import pandas as pd
# import pickle

# project_root = "/content/drive/MyDrive/FraudBehaviorEmbeddings"
# embeddings_path = os.path.join(project_root, "models/transformer/user_embeddings.pkl")

# with open(embeddings_path, 'rb') as f:
#     embeddings = pickle.load(f)

# # Convert dict of subscriber_id -> embedding to DataFrame
# df_embed = pd.DataFrame.from_dict(embeddings, orient='index')
# df_embed.reset_index(inplace=True)
# df_embed.rename(columns={'index': 'subscriber_id'}, inplace=True)

# print("✅ Embeddings Loaded:")
# df_embed.head()

# 📌 STEP 2: Convert to torch tensor
X = torch.tensor(embeddings, dtype=torch.float32)

# 📌 STEP 3: Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

model = Autoencoder(input_dim=X.shape[1], bottleneck_dim=8)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 📌 STEP 4: Train autoencoder
def train_autoencoder(X, model, optimizer, criterion, epochs=50):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

train_autoencoder(X, model, optimizer, criterion)


#  STEP 5: Save compressed embeddings
model.eval()
p, compressed_embeddings = model(X)
# print(compressed_embeddings)#==>this is tensor
compressed_np = compressed_embeddings.detach().numpy()
# print(compressed_np)-->this is array or numpy array
# When you call .detach() on a tensor:

# It returns a new tensor with the same data

# But gradients won't be tracked for it anymore

# Useful when you're not training, just using the output


# print("prediction scores=>",p)

np.save(os.path.join(root, "data/processed/compressed_embeddings.npy"), compressed_np)
print("Compressed embeddings saved!")

