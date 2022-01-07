# Model Card

## Model Details
This model is [Gradient Boosting classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) using the default hyperparameters in scikit-learn 1.0.2.

## Intended Use
The model should be used to predict if the salary of an individual exceeds $50K/yr based on census data set.

## Data
The data was obtained from the [UCI Machine Learning Repository: Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income).
The raw data set `census.csv` has 32561 rows. After cleaning, 30137 row of cleaned data in `cleaned_data.csv` was spilitted into 80% of train set, 20% of test set. To use the data for training a, One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
Evaluation metrics include precision, recall and F-beta score.
The overall performance of the model:
* Precision : 0.725
* Recall : 0.531
* F-beta : 0.613

The performance of the model given 'education' feature value is held fixed can be found in `education_slice_output.csv`

## Ethical Considerations
Model bias due to gender,racial and ethnicity discrimination in the data set should be further examined.

## Caveats and Recommendations
As the model was trained using default hyper-parameters in scikit-learn, hyper-parameter optimization and more feature engineering should be considered for better prediction results.
