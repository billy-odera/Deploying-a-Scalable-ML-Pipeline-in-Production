from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    """
    Test GET() on the root for giving a welcome message.
    """

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to Udacity MLDevops Project 3: \
                    Deploying a Machine Learning Model on Heroku with FastAPI"}

def test_post_more_than_50():
    """
    Tests POST() for a inference output less than 50k.
    """

    r = client.post("/prediction",
                           json={
                               "age": 65,
                               "fnlgt": 209280,
                               "workclass": "State-gov",
                               "education": "Masters",
                               "maritalStatus": "Married-civ-spouse",
                               "occupation": "Prof-specialty",
                               "relationship": "Husband",
                               "race": "White",
                               "sex": "Male",
                               "hoursPerWeek": 35,
                               "nativeCountry": "United-States"
                           })

    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}

def test_post_less_than_50(client):
    """
    Tests POST() for a inference output less than 50k.
    """

    r = client.post("/prediction",
                           json={
                               "age": 23,
                               "fnlgt": 263886,
                               "workclass": "Private",
                               "education": "Some-college",
                               "maritalStatus": "Never-married",
                               "occupation": "Sales",
                               "relationship": "Not-in-family",
                               "race": "Black",
                               "sex": "Female",
                               "hoursPerWeek": 20,
                               "nativeCountry": "United-States"
                           })

    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}
