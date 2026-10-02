import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from mlflow.tracking import MlflowClient
from data import load_and_split_data

def main():
    # 1. Setup MLflow Tracking with SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Wine-Cultivar-Classification")

    # 2. Load Data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # 3. Define Models and Hyperparameter Grid
    models = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": [
                {"n_estimators": 10, "max_depth": 3},
                {"n_estimators": 50, "max_depth": 5},
                {"n_estimators": 100, "max_depth": None}
            ]
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": [
                {"n_estimators": 50, "learning_rate": 0.05},
                {"n_estimators": 100, "learning_rate": 0.1},
                {"n_estimators": 150, "learning_rate": 0.2}
            ]
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_f1 = -1
    best_run_id = None

    # 4. Run Experiments
    for model_name, config in models.items():
        base_model = config["model"]
        for params in config["params"]:
            with mlflow.start_run(run_name=f"{model_name}_run"):
                model = base_model.set_params(**params)
                
                # Log Parameters
                mlflow.log_params(params)
                mlflow.log_param("model_type", model_name)

                # 5-fold Cross Validation
                cv_results = cross_validate(
                    model, X_train, y_train, cv=cv, 
                    scoring=('f1_macro', 'accuracy', 'neg_log_loss')
                )
                
                val_f1 = cv_results['test_f1_macro'].mean()
                val_acc = cv_results['test_accuracy'].mean()
                val_log_loss = -cv_results['test_neg_log_loss'].mean()

                # Log Metrics
                mlflow.log_metric("val_f1_macro", val_f1)
                mlflow.log_metric("val_accuracy", val_acc)
                mlflow.log_metric("val_log_loss", val_log_loss)

                # Train on full train set & Log Model
                model.fit(X_train, y_train)
                input_example = X_train.iloc[[0]]
                signature = infer_signature(X_train, model.predict(X_train))
                
                mlflow.sklearn.log_model(
                    model, "model", 
                    signature=signature, 
                    input_example=input_example
                )

                # Track best model
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_run_id = mlflow.active_run().info.run_id

    # 5. Register Champion Model
    print(f"Best run ID: {best_run_id} with F1: {best_f1}")
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, "WineClassifier")
    
    client = MlflowClient()
    client.set_registered_model_alias(
        name="WineClassifier",
        alias="champion",
        version=registered_model.version
    )
    print(f"Model registered as 'WineClassifier' (Version {registered_model.version}) with alias 'champion'")

if __name__ == "__main__":
    main()