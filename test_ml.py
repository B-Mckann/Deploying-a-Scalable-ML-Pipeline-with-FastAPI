import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

def test_one_model_returns_randomforest():
    """
    Tests that train_model returns a RandomForestClassifier 
    instance, confirming the expected model type is being used.
    """
    X_train = np.array([[1,2],[3,4],[5,6],[7,8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "train_model should return an instance of RandomForestClassifier"
    pass



def test_two_compute_model_metrics():
    """
    Tests that compute_model_metrics returns precision, recall, and f1
    values given known labels and predictions.
    """
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 0])
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0,abs=1e-4)
    assert recall == pytest.approx(0.6667,abs=1e-4)
    assert f1 == pytest.approx(0.8,abs=1e-4)
    pass



def test_three_inference_returns_np_array():
    """
    Test the inference function returns a numpy array of 
    predictions with the same length as the input data.
    """
    X_train = np.array([[1,2],[3,4],[5,6],[7,8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    X_test = np.array([[1,2],[3,4]])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)
    pass
