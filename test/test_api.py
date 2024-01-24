def test_get(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == 'Welcome!'


def test_upper_class(client):

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

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_lower_class(client):

    data = {
        "workclass": "Private",
        "education": "Some-college",
        "marital_status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
