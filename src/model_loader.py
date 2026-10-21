from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from .config import MODEL_DIR, MODEL_FILES, SOURCE_MODELS

class WasteCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.20), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def _checkpoint_path(filename):
    local = MODEL_DIR / filename
    if local.is_file():
        return local
    if SOURCE_MODELS is not None:
        original = SOURCE_MODELS / filename
        if original.is_file():
            return original
    raise FileNotFoundError(f"No se encontró {filename}. Colócalo en {MODEL_DIR}.")

def _state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        state = payload
    else:
        raise ValueError("Checkpoint no reconocido: se esperaba model_state_dict o un state_dict plano.")
    if not isinstance(state, dict):
        raise ValueError("model_state_dict no contiene un diccionario de pesos.")
    # DataParallel may prefix every key; normalize only that known wrapper prefix.
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state

def _build(name):
    if name == "CNN from scratch":
        return WasteCNN(10)
    if name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        return model
    if name == "Vision Transformer":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 10)
        return model
    raise KeyError(name)

@lru_cache(maxsize=3)
def load_model(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = _checkpoint_path(MODEL_FILES[name])
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only
        payload = torch.load(path, map_location="cpu")
    model = _build(name)
    try:
        model.load_state_dict(_state_dict(payload), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Pesos incompatibles para {name} ({path.name}). Verifica la arquitectura y versión de torchvision del entrenamiento. Detalle: {exc}") from exc
    model.to(device).eval()
    return model, device
