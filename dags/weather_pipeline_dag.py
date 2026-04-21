from __future__ import annotations

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="weather_stgnn_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["weather", "dlops", "stgnn"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_data",
        bash_command="python -m src.data_ingestion",
    )

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="python -m src.preprocess",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="python -m src.train",
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="python -m src.evaluate",
    )

    ingest >> preprocess >> train >> evaluate