import time
import torch
from PIL import Image
from torchvision import transforms
from .config import MODEL_INPUTS
from .model_loader import load_model

def predict(image: Image.Image, model_name: str, class_names: list[str], top_k=3):
    size, mean, std = MODEL_INPUTS[model_name]
    preprocess = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    model, device = load_model(model_name)
    tensor = tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape != (1, len(class_names)):
            raise RuntimeError(f"Salida inesperada de {model_name}: {tuple(logits.shape)}; esperada (1, {len(class_names)}).")
        probabilities = torch.softmax(logits, dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    values, indexes = probabilities.topk(min(top_k, len(class_names)))
    return {"model": model_name, "prediction": class_names[int(indexes[0])], "confidence": float(values[0]), "top": [{"class": class_names[int(i)], "probability": float(v)} for v, i in zip(values, indexes)], "inference_ms": elapsed_ms, "device": str(device)}
