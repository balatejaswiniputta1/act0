# Spatial-Temporal GNN for Weather Forecasting with DLOps

## Objective
Forecast temperature for multiple Atlanta-area stations at 1h, 6h, 12h, and 24h horizons using a spatial-temporal GNN. The workflow includes Open-Meteo ingestion, preprocessing, graph construction, model training, evaluation, DVC tracking, MLflow logging, and Airflow orchestration.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e .