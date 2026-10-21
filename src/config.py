from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_source = os.environ.get("VANTA_SOURCE_MODELS")
SOURCE_MODELS = Path(_source) if _source else None
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODEL_DIR / "comparison_results.csv"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
MODEL_FILES = {
    "CNN from scratch": "cnn_scratch_best.pth",
    "MobileNetV2": "mobilenetv2_transfer_best.pth",
    "Vision Transformer": "vit_b16_transfer_best.pth",
}
MODEL_INPUTS = {
    "CNN from scratch": (128, [0.5] * 3, [0.5] * 3),
    "MobileNetV2": (224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "Vision Transformer": (224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

def load_class_names():
    path = CLASS_NAMES_PATH
    if not path.is_file() and SOURCE_MODELS is not None:
        path = SOURCE_MODELS / "class_names.json"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró class_names.json en {CLASS_NAMES_PATH}")
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) != 10 or not all(isinstance(x, str) for x in names):
        raise ValueError("class_names.json debe contener la lista ordenada de diez clases.")
    return names
