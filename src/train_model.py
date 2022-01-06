# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_models, cat_features
from ml.slices import slice_category

# Add code to load in the data.
data = pd.read_csv('../data/cleaned_data.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

# Train a model
trained_model = train_model(X_train, y_train)

# Save the trained ML model, encoder and label binarizer
save_models(trained_model, encoder, lb)

# Get the overall performance metrics
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    encoder=encoder, lb=lb, training=False)

prediction = inference(trained_model, X_test)
prc, rcl, fb = compute_model_metrics(y_test, prediction)

print("Overall Performance Metrics of the Trained Model:")
print(f"Precision: {prc:.3f}")
print(f"Recall: {rcl:.3f}")
print(f"FBeta: {fb:.3f}")

# We fix the 'education' feature values and compute the performance metrics
cat = "education"
print(f"Performance metrics for each distinct values of {cat} feature:")
slice_category(test, cat, cat_features, trained_model, encoder, lb)
