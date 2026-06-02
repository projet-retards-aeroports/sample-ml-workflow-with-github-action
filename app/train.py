def run_training(args):
    start_time = time.time()
        
    df = load_data(DATA_URL)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    pipe = create_pipeline()
    model = train_model(pipe, X_train, y_train, {
        "Random_Forest__n_estimators": [args.n_estimators],
        "Random_Forest__criterion": [args.criterion]
    })

    best_score = model.best_score_
    test_score = model.score(X_test, y_test)

    print(f"📊 Train CV Score: {best_score:.4f}")
    print(f"📊 Test Score: {test_score:.4f}")

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_metric("train_cv_score", best_score)
    mlflow.log_metric("test_score", test_score)
    mlflow.log_metric("training_time", time.time() - start_time)

    mlflow.sklearn.log_model(
        sk_model=model.best_estimator_,
        artifact_path="model",
        registered_model_name="random_forest_regressor"
    )

    print("✅ Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=20)
    parser.add_argument("--criterion", type=str, default="squared_error")
    parser.add_argument("--experiment_name", type=str, default="california_housing")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name)

    print(f"🚀 Starting MLflow Run in experiment: {args.experiment_name}")

    # On ne fait PLUS de with mlflow.start_run()
    # MLflow gère déjà le run grâce à docker_env dans le MLproject
    run_training(args)
