from src.data import load_and_split_data

def test_data_shape():
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert X_train.shape[1] == 13
    assert len(X_train) + len(X_test) == 178