import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import inference,load_models, cat_features

# Load the data
data = pd.read_csv('../data/cleaned_data.csv')

# Load the saved model
trained_model, encoder, lb = load_models()

# Get the test data
_, test = train_test_split(data, test_size=0.20)

# Get the X_test
X_test, _, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    encoder=encoder, lb=lb, training=False)

def test_for_nulls(data):
    """
    Test to make sure there are no missing values in the cleaned_data
    """
    assert data.shape == data.dropna().shape

def test_inference(trained_model, X_test):
    """
    Test the type and value return by inference() is correct
    """
    prediction = inference(trained_model, X_test)

    assert isinstance(prediction, np.ndarray)
    assert list(np.unique(prediction)) == [0, 1]

def test_save_models(path="../model"):
    """
    Test the saved trained_model, encoder and lb files exists
    """

    assert os.path.isfile(f"{path}/trained_model.joblib")
    assert os.path.isfile(f"{path}/encoder.joblib")
    assert os.path.isfile(f"{path}/lb.joblib")
