from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from src.dataset import WeatherDataset
from src.model import STGNN, gaussian_nll_loss
from src.utils import get_device, load_yaml, set_seed, ensure_dir


def evaluate(model, loader, adj, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mean, logvar = model(xb, adj)
            loss = gaussian_nll_loss(mean, logvar, yb)
            losses.append(loss.item())
    return float(np.mean(losses))

def main() -> None:
    params = load_yaml("params.yaml")
    set_seed(params["train"]["seed"])
    ensure_dir("models")

    ds = np.load("data/processed/dataset.npz")
    x_train, y_train = ds["x_train"], ds["y_train"]
    x_val, y_val = ds["x_val"], ds["y_val"]
    adj = torch.tensor(ds["adj"], dtype=torch.float32)

    train_ds = WeatherDataset(x_train, y_train)
    val_ds = WeatherDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=params["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["train"]["batch_size"], shuffle=False)

    device = get_device(params["train"]["device"])
    adj = adj.to(device)

    model = STGNN(
        num_features=len(params["data"]["feature_cols"]),
        num_horizons=len(params["data"]["horizons"]),
        hidden_dim=params["model"]["hidden_dim"],
        gcn_out_dim=params["model"]["gcn_out_dim"],
        dropout=params["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["train"]["learning_rate"],
        weight_decay=params["train"]["weight_decay"],
    )

    mlflow.set_experiment("stgnn-weather-forecasting")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": params["train"]["epochs"],
            "batch_size": params["train"]["batch_size"],
            "lr": params["train"]["learning_rate"],
            "input_window": params["data"]["input_window"],
            "horizons": str(params["data"]["horizons"]),
        })

        best_val = float("inf")
        patience = 3
        no_improve_epochs = 0
        for epoch in range(params["train"]["epochs"]):
            model.train()
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                mean, logvar = model(xb, adj)
                loss = gaussian_nll_loss(mean, logvar, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            train_loss = float(np.mean(losses))
            val_loss = evaluate(model, val_loader, adj, device)
            mlflow.log_metric("train_nll", train_loss, step=epoch)
            mlflow.log_metric("val_nll", val_loss, step=epoch)
            print(f"Epoch {epoch+1}: train_nll={train_loss:.4f}, val_nll={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                no_improve_epochs = 0
                torch.save(model.state_dict(), "models/best_model.pt")
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        mlflow.log_artifact("models/best_model.pt")


if __name__ == "__main__":
    main()