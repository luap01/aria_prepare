
import json
import numpy as np
import pandas as pd

def save_transform_json(path, T, meta=None):
    payload = {"T_A_from_O": T.tolist()}
    if meta:
        payload["meta"] = meta
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def save_metrics_json(path, metrics: dict):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def save_metrics_csv(path, metrics: dict):
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (list, tuple, np.ndarray, dict)):
            continue
        flat[k] = v
    df = pd.DataFrame([flat])
    df.to_csv(path, index=False)
