from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.graph import build_adjacency_matrix, normalize_adjacency
from src.utils import ensure_dir, load_yaml


def create_windows(data: np.ndarray, input_window: int, horizons: list[int], target_feature_idx: int) -> tuple[np.ndarray, np.ndarray]:
    max_h = max(horizons)
    xs, ys = [], []
    total_steps = data.shape[0]
    for t in range(input_window, total_steps - max_h):
        x = data[t - input_window:t]
        y = []
        for h in horizons:
            y.append(data[t + h, :, target_feature_idx])
        y = np.stack(y, axis=0)
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main() -> None:
    ensure_dir("data/processed")
    params = load_yaml("params.yaml")
    stations_cfg = load_yaml("configs/stations.yaml")

    df = pd.read_csv("data/raw/weather_raw.csv", parse_dates=["time"])
    feature_cols = params["data"]["feature_cols"]
    target_col = params["data"]["target_col"]

    df = df.sort_values(["time", "station"]).copy()
    df[feature_cols] = df.groupby("station")[feature_cols].ffill().bfill()

    station_names = [s["name"] for s in stations_cfg["stations"]]
    pivot_frames = []
    for station in station_names:
        station_df = df[df["station"] == station].sort_values("time")
        pivot_frames.append(station_df[feature_cols].to_numpy())

    data = np.stack(pivot_frames, axis=1)
    t, n, f = data.shape
    flat = data.reshape(-1, f)

    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    data_scaled = flat_scaled.reshape(t, n, f)

    target_idx = feature_cols.index(target_col)
    x, y = create_windows(
        data=data_scaled,
        input_window=params["data"]["input_window"],
        horizons=params["data"]["horizons"],
        target_feature_idx=target_idx,
    )

    total = len(x)
    train_end = int(total * params["data"]["train_ratio"])
    val_end = int(total * (params["data"]["train_ratio"] + params["data"]["val_ratio"]))

    stations = stations_cfg["stations"]
    adj = build_adjacency_matrix(stations, params["graph"]["distance_threshold_km"])
    adj_norm = normalize_adjacency(adj)

    np.savez_compressed(
        "data/processed/dataset.npz",
        x_train=x[:train_end],
        y_train=y[:train_end],
        x_val=x[train_end:val_end],
        y_val=y[train_end:val_end],
        x_test=x[val_end:],
        y_test=y[val_end:],
        adj=adj_norm.astype(np.float32),
    )
    joblib.dump(scaler, "data/processed/scaler.joblib")
    print("Saved processed dataset and scaler.")


if __name__ == "__main__":
    main()