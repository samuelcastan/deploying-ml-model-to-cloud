import pandas as pd
import joblib
import pytest


@pytest.fixture(scope="session")
def expected_labels():
    expected_labels = ["<=50K", ">50K"]
    return expected_labels

@pytest.fixture(scope="session")
def features():

    features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]

    return features


@pytest.fixture(scope="session")
def data():
    data_path = "data/clean/census.csv"

    if data_path is None:
        pytest.fail(
            "You must provide the clean_data path on the conftest.py file")

    data = pd.read_csv(data_path)

    return data


@pytest.fixture(scope="session")
def pipeline():
    pipeline_path = "model/inference_pipeline.pkl"

    if pipeline_path is None:
        pytest.fail(
            "You must provide the inference_pipeline path on the conftest.py file")

    pipeline = joblib.load(pipeline_path)

    return pipeline
