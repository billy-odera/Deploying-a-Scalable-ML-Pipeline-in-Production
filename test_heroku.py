'''
Script that uses the requests module to do one POST on the live API @ https://p3-prediction-app.herokuapp.com/
'''

import requests


data = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 65278,
    "education": "HS-grad",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
}

r = requests.post('https://p3-prediction-app.herokuapp.com/prediction',
                  json=data)

assert r.status_code == 200

print(f"Response code: {r.status_code}")
print(f"Response body: {r.json()}")
