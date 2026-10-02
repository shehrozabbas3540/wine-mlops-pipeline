import mlflow
import time
import pytest
from src.data import load_and_split_data
from sklearn.metrics import f1_score

@pytest.fixture
def setup():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    _, X_test, _, y_test = load_and_split_data()
    model = mlflow.pyfunc.load_model("models:/WineClassifier@champion")
    return model, X_test, y_test

def test_metric_threshold(setup):
    model, X_test, y_test = setup
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    assert f1 >= 0.88, f"F1 score {f1} is below 0.88"

def test_inference_latency(setup):
    model, X_test, _ = setup
    start_time = time.time()
    model.predict(X_test)
    latency = (time.time() - start_time) * 1000
    assert latency <= 30, f"Latency {latency} ms exceeded 30ms limit"

def test_output_schema(setup):
    model, X_test, _ = setup
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1, 2}), "Invalid classes in output"