import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def test_inferencing(data, features, pipeline):

    X = data[features]

    pred = pipeline.predict(X)

    assert isinstance(pred, np.ndarray)


def test_predicted_labels(data, features, pipeline, expected_labels):

    X = data[features]

    predictions = pipeline.predict(X)

    labels = pd.Series(predictions).unique()

    assert np.array_equal(labels, expected_labels)


def test_overall_balanced_accuracy(data, features, pipeline):

    y = data["salary"]
    y_pred = pipeline.predict(data[features])
    score = balanced_accuracy_score(y_true=y, y_pred=y_pred)

    assert score >= .70
    