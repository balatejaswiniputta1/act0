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
```

## Reproduce the Pipeline
```bash
python -m src.data_ingestion
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

Or run the tracked DVC workflow:
```bash
dvc repro
```

## Presentation Outputs
Evaluation writes poster- and dashboard-friendly artifacts under `reports/`:

- `reports/metrics.json`: DVC-friendly metrics with errors in degrees Celsius and normalized units.
- `reports/tables/horizon_metrics.csv`: MAE/RMSE/anomaly correlation for 1h, 6h, 12h, and 24h horizons.
- `reports/tables/station_metrics.csv`: station-level MAE/RMSE and per-horizon station MAE.
- `reports/tables/baseline_comparison.csv`: ST-GNN comparison against persistence and LSTM-only baselines.
- `reports/experiments/sweep_results.csv`: compact tuning sweep ranked by validation MAE.
- `reports/predictions/sample_predictions.csv`: compact table of actual vs predicted temperatures.
- `reports/plots/`: actual-vs-predicted traces, horizon bars, station bars, loss curve, and poster summary.
- `reports/best_run_summary.md`: final configuration, validation result, and test-set interpretation.
- `reports/POSTER_CONTENT.md`: concise poster-ready research narrative.

## Local Dashboard
After generating reports, launch the demo dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard summarizes the research story, ST-GNN methodology, DVC/MLflow/Airflow workflow, metrics, plots, and sample predictions.
