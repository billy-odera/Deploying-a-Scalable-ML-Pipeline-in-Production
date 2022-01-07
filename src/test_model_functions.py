import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.ml.data import process_data
from src.ml.model import inference,load_models, cat_features

def load_data():
    """
    Load preprocessed data and trained model
    """
    data = pd.read_csv('data/cleaned_data.csv')
    trained_model, encoder, lb = load_models('model')
    _, test = train_test_split(data, test_size=0.20)

    X_test, _, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        encoder=encoder, lb=lb, training=False)

    return data, trained_model, X_test

def test_for_nulls():
    """
    Test to make sure there are no missing values in the cleaned_data
    """
    data, _, _ = load_data()
    assert data.shape == data.dropna().shape

def test_inference():
    """
    Test the type and value return by inference() is correct
    """
    _, trained_model, X_test = load_data()
    prediction = inference(trained_model, X_test)

    assert isinstance(prediction, np.ndarray)
    assert list(np.unique(prediction)) == [0, 1]

def test_save_models(path="model"):
    """
    Test the saved trained_model, encoder and lb files exists
    """

    assert os.path.isfile(f"{path}/trained_model.joblib")
    assert os.path.isfile(f"{path}/encoder.joblib")
    assert os.path.isfile(f"{path}/lb.joblib")


test_for_nulls()
test_inference()
test_save_models()
