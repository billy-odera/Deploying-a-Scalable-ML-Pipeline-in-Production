from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, n_estimators=100):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    GBClassifier = GradientBoostingClassifier(n_estimators)
    GBClassifier.fit(X_train, y_train)

    return GBClassifier


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    prediction = model.predict(X)

    return prediction


def save_models(trained_model, encoder, lb, path="../model"):
    """
    Save the trained model, encoder and LabelBinarizer in the specified path.

    Inputs
    ------
    trained_model:
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer
    path: str
        The location to save the models (default="../model")
    """

    dump(trained_model,f"{path}/trained_model.joblib")
    dump(encoder,f"{path}/encoder.joblib")
    dump(lb,f"{path}/lb.joblib")


def load_models(path="model"):
    """
    Load the trained model, encoder and LabelBinarizer stored in the specified path.

    Inputs
    ------
    path: str
        The location that store the models (default="../model")

    Returns
    ------
    trained_model:
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer
    """

    trained_model = load(f"{path}/trained_model.joblib")
    encoder = load(f"{path}/encoder.joblib")
    lb = load(f"{path}/lb.joblib")

    return trained_model, encoder, lb
