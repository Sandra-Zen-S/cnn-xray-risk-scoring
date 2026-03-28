import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

from utils.dataset import ChestXrayDataset
from utils.preprocess import get_transforms

# -------------------------
# MODEL
# -------------------------
def get_model():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )

    return model

# -------------------------
# DATA
# -------------------------
dataset = ChestXrayDataset(
    "data/labels.csv",
    "data/images",
    transform=get_transforms()
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------
# TRAINING SETUP
# -------------------------
model = get_model()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# TRAIN LOOP
# -------------------------
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "models/model.pth")
print("Model saved successfully!")