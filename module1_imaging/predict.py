import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from utils.preprocess import get_transforms

# -------------------------
# MODEL
# -------------------------
def get_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model

def load_model():
    model = get_model()
    model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
    model.eval()
    return model

# -------------------------
# PREDICTION
# -------------------------
def predict_image(image, model):
    transform = get_transforms()
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        score = model(image).item()

    return score

# -------------------------
# TRIAGE
# -------------------------
def get_priority(score):
    if score > 0.50:
        return "HIGH"
    elif score > 0.30:
        return "MEDIUM"
    else:
        return "LOW"