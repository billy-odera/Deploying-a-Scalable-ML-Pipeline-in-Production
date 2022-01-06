
import os
from pandas.core.frame import DataFrame
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from typing import Literal
from src.ml.data import process_data
from src.ml.model import compute_model_metrics, inference, load_models, cat_features

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Adult(BaseModel):
    """
    Class to ingest the body from POST
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    maritalStatus: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hoursPerWeek: int
    nativeCountry: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "maritalStatus": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "hoursPerWeek": "40",
                "nativeCountry": "United-States"
            }
        }

# GET on the root giving a welcome message
@app.get("/")
async def get_items():
    return {"greeting": "Welcome to Udacity MLDevops Project 3: Deploying a Machine Learning Model on Heroku with FastAPI"}

# POST that does model inference
@app.post("/prediction")
async def make_inference(input_data: Adult):
    # Load the models
    trained_model, encoder, lb = load_models("model")

    array = np.array([[
        input_data.age, input_data.workclass, input_data.fnlgt,
        input_data.education, input_data.maritalStatus,
        input_data.occupation, input_data.relationship, input_data.race,
        input_data.sex, input_data.hoursPerWeek, input_data.nativeCountry
    ]])

    df = DataFrame(data=array,
                   columns=[
                       "age",
                       "workclass",
                       "fnlgt",
                       "education",
                       "marital-status",
                       "occupation",
                       "relationship",
                       "race",
                       "sex",
                       "hours-per-week",
                       "native-country",
                   ])

    X, _, _, _ = process_data(df,
                              categorical_features=cat_features,
                              encoder=encoder,
                              lb=lb,
                              training=False)

    # prediction = trained_model.predict(X)
    prediction = inference(trained_model, X)
    y = lb.inverse_transform(prediction)[0]

    return {"prediction": y}
