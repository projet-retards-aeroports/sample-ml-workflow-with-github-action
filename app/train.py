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
import mlflow
import mlflow.catboost
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
# 
load_dotenv()

# ====================== S3 LOADING ======================
s3 = boto3.client('s3')

def load_from_s3(folder: str, filename: str) -> bytes | None:
    """Download file from S3"""
    key = f"projet_final_lead/{folder}/{filename}"
    try:
        response = s3.get_object(Bucket=os.getenv("BUCKET"), Key=key)
        print(f"S3 <- {key}")
        return response['Body'].read()
    except Exception as e:
        print(f"Erreur S3 {key} : {e}")
        return None


# ====================== CONFIGURATION MLFLOW ======================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("pl_retards_vols")
print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}\n")


def train_model(df, model_name: str, run_id: str,
                iterations: int = 300,
                learning_rate: float = 0.055,
                depth: int = 8,
                loss_function: str = "RMSE",
                eval_metric: str = "RMSE",
                random_seed: int = 42,
                early_stopping_rounds: int = 300,
                task_type: str = "CPU",
                l2_leaf_reg: float = 3,
                random_strength: float = 1.0,
                bagging_temperature: float = 0.7):

    with mlflow.start_run(run_name=f"{model_name}*{run_id}") as run:
        print(f"\n=== Entraînement {model_name} ===")

        X = df.drop(columns=["scheduled_utc", "revised_utc", "flight_number", "delay_minutes"])
        y = df["delay_minutes"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        cat_features = [
            "aeroport_depart", "aeroport_arrivee", "terminal",
            "airline_icao", "airline_name", "aircraft_model",
            "aircraft_family", "aircraft_size_category",
            "holiday_name", "period_of_day"
        ]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
            eval_metric=eval_metric,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
            verbose=1000,
            task_type=task_type,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # Log parameters
        mlflow.log_params({
            **model.get_params(),
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": model.get_best_iteration()
        })

        # Evaluation
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)

        mlflow.log_metric("MAE", round(mae, 4))
        mlflow.log_metric("RMSE", round(rmse, 4))
        mlflow.log_metric("R2", round(r2, 4))
        mlflow.log_metric("best_iteration", model.get_best_iteration())

        # Feature Importance Plot
        print("Calcul et logging du Feature Importance Plot...")
        feature_names = X_train.columns.tolist()
        importance = model.get_feature_importance()
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values(by='importance', ascending=False)

        plt.figure(figsize=(12, 10))
        sns.barplot(data=fi_df.head(20), x='importance', y='feature', palette="viridis")
        plt.title(f"Top 20 Feature Importance - {model_name}")
        plt.tight_layout()

        plot_path = f"{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=220, bbox_inches='tight')
        mlflow.log_artifact(plot_path, artifact_path="feature_importance")
        plt.close()

        # Log model
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            registered_model_name=f"CatBoost_{model_name}_Delay_Prediction"
        )

        print(f"\n=== Résultats {model_name} ===")
        print(f"MAE : {mae:.3f} minutes")
        print(f"RMSE : {rmse:.3f} minutes")
        print(f"R² : {r2:.4f}")
        print(f"Best iteration : {model.get_best_iteration()}")
        print("="*70 + "\n")

        return model


def train_pipeline(run_id: str = None, **catboost_params):
    if run_id is None or run_id == "latest":
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print(f"=== Lancement du Train Pipeline - Run ID: {run_id} ===\n")

    # Departures
    depart_bytes = load_from_s3(f"processed/train/{run_id}", f"final_departures_{run_id}.parquet")
    if depart_bytes is None:
        raise FileNotFoundError(f"Could not load departures file for run_id: {run_id}")
    df_depart = pd.read_parquet(io.BytesIO(depart_bytes))
    train_model(df_depart, "Departure", run_id, **catboost_params)

    # Arrivals
    arrive_bytes = load_from_s3(f"processed/train/{run_id}", f"final_arrivals_{run_id}.parquet")
    if arrive_bytes is None:
        raise FileNotFoundError(f"Could not load arrivals file for run_id: {run_id}")
    df_arrive = pd.read_parquet(io.BytesIO(arrive_bytes))
    train_model(df_arrive, "Arrival", run_id, **catboost_params)

    print(f"\n=== Train Pipeline terminé ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)

    # CatBoost arguments
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.055)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--loss_function", type=str, default="RMSE")
    parser.add_argument("--eval_metric", type=str, default="RMSE")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--early_stopping_rounds", type=int, default=300)
    parser.add_argument("--task_type", type=str, default="CPU")
    parser.add_argument("--l2_leaf_reg", type=float, default=3)
    parser.add_argument("--random_strength", type=float, default=1.0)
    parser.add_argument("--bagging_temperature", type=float, default=0.7)

    args = parser.parse_args()

    catboost_params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "loss_function": args.loss_function,
        "eval_metric": args.eval_metric,
        "random_seed": args.random_seed,
        "early_stopping_rounds": args.early_stopping_rounds,
        "task_type": args.task_type,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "bagging_temperature": args.bagging_temperature,
    }

    train_pipeline(args.run_id, **catboost_params)
