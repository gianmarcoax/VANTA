from pathlib import Path
import pandas as pd
from .config import METRICS_PATH

METRIC_COLUMNS = {"accuracy": "Accuracy", "precision_macro": "Precision macro", "recall_macro": "Recall macro", "f1_macro": "F1 macro"}

def read_metrics(path: Path = METRICS_PATH):
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {"model", *METRIC_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError(f"El CSV de métricas no contiene estas columnas: {sorted(required - set(frame.columns))}")
    return frame
