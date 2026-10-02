import mlflow
import time
from data import load_and_split_data
from sklearn.metrics import f1_score, accuracy_score

def evaluate_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    _, X_test, _, y_test = load_and_split_data()

    # Load Champion Model
    model_uri = "models:/WineClassifier@champion"
    model = mlflow.pyfunc.load_model(model_uri)

    # Predict and calculate latency
    start_time = time.time()
    preds = model.predict(X_test)
    latency_ms = (time.time() - start_time) * 1000

    f1 = f1_score(y_test, preds, average='macro')
    print(f"Test F1: {f1:.4f}")
    print(f"Batch Inference Time: {latency_ms:.2f} ms")

if __name__ == "__main__":
    evaluate_model()