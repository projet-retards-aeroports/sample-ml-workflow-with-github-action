import argparse
import pandas as pd
import time
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
# train 11&&&&&&&&&&&1&a&&1
# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def load_data(url):
    """
    Load dataset from the given URL.
    """
    try:
        df = pd.read_csv(url)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into X (features) and y (target).
    """
    print("⚙️ Preprocessing data...")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_pipeline():
    """
    Create a machine learning pipeline with StandardScaler and RandomForestRegressor.
    """
    return Pipeline(steps=[
        ("standard_scaler", StandardScaler()),
        ("Random_Forest", RandomForestRegressor())
    ])

def train_model(pipe, X_train, y_train, param_grid, cv=2):
    """
    Train the model using GridSearchCV.
    """
    print(f"🏋️ Training model with grid: {param_grid}")
    # verbose=0 to keep logs clean in CI/CD
    model = GridSearchCV(pipe, param_grid, verbose=0, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    return model

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Parse ONLY Model Hyperparameters
    parser = argparse.ArgumentParser(description="Random Forest Training Script")
    parser.add_argument("--n_estimators", type=int, default=20)
    parser.add_argument("--criterion", type=str, default="squared_error")
    parser.add_argument("--experiment_name", type=str, default="california_housing")
    args = parser.parse_args()
    
    # Optional: If you want to force using ID, you would grab MLFLOW_EXPERIMENT_ID
    # But as discussed, Name is safer for portability.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name)
    
    print(f"🚀 Starting MLflow Run in experiment: {args.experiment_name}")

    # 3. Configuration
    DATA_URL = "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"
    
    param_grid = {
        "Random_Forest__n_estimators": [args.n_estimators],
        "Random_Forest__criterion": [args.criterion]
    }

    # 4. Start Run
    with mlflow.start_run():
        start_time = time.time()

        # Load & Preprocess
        df = load_data(DATA_URL) # Ensure load_data is defined in your script
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # Train
        pipe = create_pipeline() # Ensure create_pipeline is defined
        model = train_model(pipe, X_train, y_train, param_grid)

        # Logging
        best_score = model.best_score_
        test_score = model.score(X_test, y_test)
        
        print(f"📊 Train CV Score: {best_score:.4f}")
        print(f"📊 Test Score:     {test_score:.4f}")

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("criterion", args.criterion)
        
        mlflow.log_metric("train_cv_score", best_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("training_time", time.time() - start_time)

        print("💾 Saving model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model.best_estimator_,
            artifact_path="model",
            registered_model_name="random_forest_regressor"
        )
        
        print("✅ Training Complete.")
