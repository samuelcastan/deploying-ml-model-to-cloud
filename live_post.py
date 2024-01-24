import requests

data = {
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native_country": "United-States"
}

request = requests.post(
    "https://predicting-salary.onrender.com/predict",
    json=data)

print(request.status_code)
print(request.text)
