# Regional Weather Forecasting with a Spatial-Temporal Graph Neural Network

## Motivation
Regional weather changes are spatially connected: nearby stations often share fronts, humidity patterns, and pressure systems. A spatial-temporal graph neural network (ST-GNN) can model both the time history at each station and the relationships between nearby stations.

## Problem Statement
Forecast hourly temperature for multiple Atlanta-area weather stations at 1h, 6h, 12h, and 24h horizons using recent meteorological observations and station geography.

## Methodology
The pipeline ingests hourly Open-Meteo weather observations, preprocesses station-level features, constructs a graph from geographic station distances, trains an ST-GNN, and evaluates forecasts in original temperature units. The model receives a 24-hour input window and predicts temperature for all stations across four forecast horizons.

## Architecture
Stations are represented as graph nodes. Edges connect stations within a configurable distance threshold. Each hourly station snapshot passes through graph convolution layers, and the resulting station embeddings are fed into an LSTM to capture temporal dynamics. Separate output heads predict the forecast mean and log variance for probabilistic training.

## Pipeline Overview
Data ingestion: downloads hourly temperature, humidity, wind, and pressure features for configured stations.

Preprocessing: fills missing values, standardizes features, creates supervised windows, and saves the normalized adjacency matrix.

Training: trains the ST-GNN with Gaussian negative log likelihood and logs metrics/artifacts to MLflow.

Evaluation: converts predictions back to degrees Celsius, computes per-horizon and per-station metrics, and generates visual reports.

DLOps: DVC tracks data/model stages, MLflow records experiments, and Airflow orchestrates the end-to-end workflow.

## Experiments
The current experiment uses a 24-hour input window and evaluates four forecast horizons: 1h, 6h, 12h, and 24h. Performance is reported with MAE, RMSE, and anomaly correlation, with MAE/RMSE converted back to degrees Celsius for interpretability.

## Results
Use `reports/tables/horizon_metrics.csv` for the final values on the poster. Emphasize:

- Short-horizon forecasts are expected to be most accurate because atmospheric persistence is strongest.
- Longer horizons show how forecast uncertainty and error grow over time.
- Station-level metrics demonstrate whether the model generalizes evenly across the region.
- The actual-vs-predicted plots show that the ST-GNN follows daily temperature cycles rather than only reporting aggregate metrics.

## Conclusion
The project demonstrates a complete research and DLOps workflow for regional temperature forecasting. The ST-GNN combines spatial station relationships with temporal weather history, while DVC, MLflow, and Airflow make the workflow reproducible and demo-ready.

## Future Work
Add additional weather stations, include longer date ranges and seasonal variation, compare against baseline models such as persistence and LSTM-only forecasts, tune graph thresholds, and calibrate probabilistic uncertainty estimates.

## Suggested Poster Takeaway
This project shows how graph-based deep learning can turn a network of weather stations into a reproducible, interpretable multi-horizon forecasting system.
