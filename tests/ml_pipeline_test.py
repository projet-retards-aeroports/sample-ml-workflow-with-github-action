import pytest
import pandas as pd
import numpy as np
import os
import boto3
from botocore.exceptions import ClientError

from app.train import train_model
import argparse
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import mlflow                          # ← Make sure this is here
import mlflow.catboost                 # ← Make sure this is here
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

from src.pipelines.custom_libs.load_from import load_from_s3


# ==========================================
# DEFAULT RUN_ID
# ==========================================
DEFAULT_RUN_ID = "2026-05-25_124326_039dd1"


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def mock_df():
    """Small realistic DataFrame matching your airport data."""
    np.random.seed(42)
    n = 40

    data = {
        'scheduled_utc': pd.date_range('2023-01-01', periods=n, freq='H'),
        'revised_utc': pd.date_range('2023-01-01', periods=n, freq='H') 
                       + pd.to_timedelta(np.random.randint(0, 180, n), unit='m'),
        'flight_number': [f'AF{1000 + i}' for i in range(n)],
        'delay_minutes': np.random.randint(-20, 160, n),
        'aeroport_depart': np.random.choice(['CDG', 'ORY', 'NCE', 'LYS'], n),
        'aeroport_arrivee': np.random.choice(['CDG', 'ORY', 'NCE', 'LYS'], n),
        'terminal': np.random.choice(['1', '2', '3', '4'], n),
        'airline_icao': np.random.choice(['AFR', 'EZY', 'RYR'], n),
        'airline_name': np.random.choice(['Air France', 'EasyJet', 'Ryanair'], n),
        'aircraft_model': np.random.choice(['A320', 'B737', 'A350'], n),
        'aircraft_family': np.random.choice(['A320 Family', 'B737 Family'], n),
        'aircraft_size_category': np.random.choice(['Narrow-body', 'Wide-body'], n),
        'holiday_name': np.random.choice(['None', 'Christmas'], n),
        'period_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening'], n),
    }
    return pd.DataFrame(data)


# ==========================================
# S3 HELPER
# ==========================================
def check_s3_files_exist(run_id: str) -> bool:
    """Check if both departure and arrival files exist on S3."""
    bucket = os.getenv("BUCKET")
    if not bucket:
        pytest.skip("BUCKET secret not available (running locally?)")

    s3 = boto3.client("s3")
    prefix = f"processed/train/{run_id}"

    required_files = [
        f"{prefix}/final_departures_{run_id}.parquet",
        f"{prefix}/final_arrivals_{run_id}.parquet"
    ]

    for key in required_files:
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"❌ File not found on S3: {key}")
                return False
            else:
                raise
    return True


# ==========================================
# TESTS
# ==========================================
def test_train_model_runs_without_error(mock_df):
    """Fast smoke test (no S3 needed)."""
    model = train_model(
        df=mock_df,
        model_name="Departure",
        run_id="test-smoke-001",
        iterations=30,
        depth=4,
        learning_rate=0.1,
        task_type="CPU",
    )
    assert model is not None


def test_s3_files_exist_real():
    """
    Real S3 test.
    - Works in GitHub Actions (secrets are injected)
    - Skips gracefully if running locally without secrets
    """
    run_id = os.getenv("TEST_RUN_ID", DEFAULT_RUN_ID)

    print(f"\n🔍 Checking real S3 files for run_id: {run_id}")

    exists = check_s3_files_exist(run_id)

    assert exists is True, (
        f"❌ Files not found on S3 for run_id='{run_id}'.\n"
        f"Expected files:\n"
        f"  - processed/train/{run_id}/final_departures_{run_id}.parquet\n"
        f"  - processed/train/{run_id}/final_arrivals_{run_id}.parquet"
    )
