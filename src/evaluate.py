from __future__ import annotations

import json
import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dataset import WeatherDataset
from src.model import STGNN
from src.utils import ensure_dir, get_device, load_yaml


def anomaly_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_anom = y_true - np.mean(y_true)
    pred_anom = y_pred - np.mean(y_pred)
    numerator = np.sum(true_anom * pred_anom)
    denominator = np.sqrt(np.sum(true_anom**2) * np.sum(pred_anom**2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def main() -> None:
    ensure_dir("reports")
    params = load_yaml("params.yaml")
    ds = np.load("data/processed/dataset.npz")
    x_test, y_test = ds["x_test"], ds["y_test"]
    adj = torch.tensor(ds["adj"], dtype=torch.float32)

    device = get_device(params["train"]["device"])
    model = STGNN(
        num_features=len(params["data"]["feature_cols"]),
        num_horizons=len(params["data"]["horizons"]),
        hidden_dim=params["model"]["hidden_dim"],
        gcn_out_dim=params["model"]["gcn_out_dim"],
        dropout=params["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
    model.eval()

    xb = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        mean, logvar = model(xb, adj.to(device))

    preds = mean.cpu().numpy()
    y_true = y_test

    metrics = {}
    for idx, horizon in enumerate(params["data"]["horizons"]):
        yt = y_true[:, idx, :].reshape(-1)
        yp = preds[:, idx, :].reshape(-1)
        metrics[f"horizon_{horizon}h"] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "acc": float(anomaly_correlation_coefficient(yt, yp)),
        }

    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()