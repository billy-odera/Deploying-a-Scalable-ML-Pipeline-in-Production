import numpy as np
import pandas as pd
from beautifultable import BeautifulTable

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def slice_category(df, cat, cat_features, trained_model, encoder, lb):
    '''
    For a given categorical variable, this function computes the metrics when
    its value is held fixed. E.g. for education, it would print out the model
    metrics for each slice of data that has a particular value for education.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    cat: str
        Name of the feature to be held fixed
    categorical_features: list[str]
        List containing the names of the categorical features
    trained_model:
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer

    Returns
    -------
    None

    Output:
    -------
    The model metrics for each value of the feature will be save as 
    {cat}_slice_output.csv under '/slices' folder. The table will also be
    printed out.

    '''

    # Initiate a table
    table = BeautifulTable()
    table.rows.append([f"Feature:{cat}","Precision", "Recall", "FBeta"])

    for cls in df[cat].unique():
        df_temp = df[df[cat] == cls]

        X_test, y_test, _, _ = process_data(
            df_temp,
            categorical_features=cat_features,
            label="salary", encoder=encoder, lb=lb, training=False)

        prediction = inference(trained_model, X_test)
        prc, rcl, fb = compute_model_metrics(y_test, prediction)

        table.rows.append([cls, prc, rcl, fb])

    # Export the table as csv file
    table.to_csv(f"../slices/{cat}_slice_output.csv")

    print(table)
