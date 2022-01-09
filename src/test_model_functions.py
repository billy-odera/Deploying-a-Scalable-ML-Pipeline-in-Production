import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from src.ml.data import process_data
from src.ml.model import inference,load_models, cat_features, compute_model_metrics

def load_data():
    """
    Load preprocessed data and trained model
    """
    data = pd.read_csv('data/cleaned_data.csv')
    trained_model, encoder, lb = load_models('model')
    _, test = train_test_split(data, test_size=0.20)

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        encoder=encoder, lb=lb, training=False)

    return data, trained_model, X_test, y_test

def test_for_nulls():
    """
    Test to make sure there are no missing values in the cleaned_data
    """
    data, _, _, _ = load_data()
    assert data.shape == data.dropna().shape

def test_inference():
    """
    Test the type and value return by inference() is correct
    """
    _, trained_model, X_test, _ = load_data()
    prediction = inference(trained_model, X_test)

    assert isinstance(prediction, np.ndarray)
    assert list(np.unique(prediction)) == [0, 1]

def test_trained_model():
    """
    Test the type of the trained_model is a GradientBoostingClassifier
    """

    _, trained_model, _, _= load_data()
    assert type(trained_model) == sklearn.ensemble.GradientBoostingClassifier

def test_compute_model_metrics():
    """
    Test the range of performance metrics returned by compute_model_metrics()
    """
    _, trained_model, X_test, y_test = load_data()
    prediction = inference(trained_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, prediction)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
