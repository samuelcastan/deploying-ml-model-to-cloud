"""
Trains a Random Forest pipeline on a balanced dataset and
evaluates its performance

Created by Samuel Castan
Last Updated: Jan 2024
"""

import warnings
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")


def go(args):

    # Read data
    df = pd.read_csv(args.clean_data)

    features = args.cat_features.split(",")

    # Features and target variables
    X = df[features]
    y = df[args.target]

    X_train, X_test, y_train, y_test = balance_dataset(
        X,
        y,
        test_size=args.test_size,
        features=features,
        target=args.target,
        random_state=args.random_state
    )

    # train pipeline
    pipeline = train_pipeline(
        features=features,
        X_train=X_train,
        y_train=y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        random_state=args.random_state_rf,
        n_jobs=args.n_jobs)

    # evaluate model
    model_performance(pipeline=pipeline, X_test=X_test, y_test=y_test)

    # perform data slicing
    data_slicing_evaluation(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        cat_features=features)

    # Save pipeline
    save_pipeline(pipeline=pipeline, path=args.pipeline_path)


def balance_dataset(X, y, test_size, features, target, random_state):
    """"
    Balances a dataset for training by downsampling to the
    same length of the minority class

    Inputs:
        X: Independent variables
        y: Target or dependent variable
        test_size: Percentage of the dataset to be tested on
        features: List of feature names to train on
        target: Target variable to train on
        random_state: Reproducibility number
    """

    # First split to obtain the testing dataset and temporal set for balancing
    # the classes
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Start the imbalanced dataset
    df_temp = pd.concat([X_temp, y_temp], axis=1)

    del X_temp, y_temp

    undersampling_size = len(df_temp[df_temp[target] == ">50K"])

    df_temp = pd.concat([df_temp[df_temp[target] == "<=50K"].sample(
        undersampling_size), df_temp[df_temp[target] == ">50K"]])

    X_train = df_temp[features]
    y_train = df_temp[target]

    return X_train, X_test, y_train, y_test


def train_pipeline(
        features,
        X_train,
        y_train,
        n_estimators,
        max_depth,
        max_features,
        random_state,
        n_jobs):
    """""
    Trains the entire ML inference pipeline: should train on the provided data.

    Inputs:
        features: List of feature names to train on
        X_train: Instances to train on
        y_train: Target variable
        n_estimators: Random Forest hyperparameter
        max_depth: Random Forest hyperparameter
        max_features: Random Forest hyperparameter
        random_state: Random Forest hyperparameter
        n_job: Random Forest hyperparameter

    Output:
        pipeline: Random Forest pipeline
    """

    column_transformer = ColumnTransformer(
        transformers=[
            # name, transformer, columns
            ('cat', OneHotEncoder(handle_unknown="ignore"), features),
        ],
        # Ignore numerical columns
        remainder='passthrough'
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs
    )

    pipeline = Pipeline(
        steps=[
            ('one-hot-encoder', column_transformer),
            ('classifier', classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def inference(pipeline, X_pred):
    """
    Predicts new instances using the trained pipeline

    Args:
        pipeline (Sklearn Pipeline): Trained pipeline
        X_pred (DataFrame-Numpy Array): Instance(s) to predict on

    Returns:
        y_pred (Numpy Array): Predicted classes for X_pred
    """

    y_pred = pipeline.predict(X_pred)

    return y_pred


def model_performance(pipeline, X_test, y_test, in_place=True):

    # Make predictions on the test set
    y_pred = inference(pipeline, X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    recall_50k_up = recall_score(
        y_true=y_test,
        y_pred=y_pred,
        pos_label=">50K")
    recall_50k_low = recall_score(
        y_true=y_test,
        y_pred=y_pred,
        pos_label="<=50K")
    precision_50k_up = precision_score(
        y_true=y_test, y_pred=y_pred, pos_label=">50K")
    precision_50k_low = precision_score(
        y_true=y_test, y_pred=y_pred, pos_label="<=50K")

    # Path to save metrics report
    file_path = args.classification_report

    with open(file_path, 'w') as file:
        file.write("Accuracy: {:.4f}\n".format(accuracy))
        file.write("Balanced Accuracy: {:.4f}\n".format(balanced_accuracy))
        file.write("Recall (>50K): {:.4f}\n".format(recall_50k_up))
        file.write("Recall (<=50K): {:.4f}\n".format(recall_50k_low))
        file.write("Precision (>50K): {:.4f}\n".format(precision_50k_up))
        file.write("Precision (<=50K): {:.4f}\n".format(precision_50k_low))


def data_slicing_evaluation(
        pipeline,
        X_test,
        y_test,
        cat_features=["education"]):
    """
    Performs the balanced accuracy for value slice in the categorical features

    Args:
        pipeline (Sklearn Pipeline): Trained pipeline
        X_test (DataFrame-Numpy Array): Instances to predict on
        y_test (DataFrame-Numpy Array): Target variable
        cat_features (list, optional): cateogorical features to slice on.
        Defaults to ["education"].
    """

    df_temp = pd.concat([X_test, y_test], axis=1)

    try:
        with open(args.data_slice_report, "w") as file:
            file.write("Category - Value - Balanced Acccuracy\n")
            for category in cat_features:
                for value in df_temp[category].unique():
                    X_category = df_temp[df_temp[category]
                                         == value].drop(["salary"], axis=1)
                    y_category = df_temp[df_temp[category] == value]["salary"]

                    y_pred = pipeline.predict(X_category)

                    balaced_accuracy = balanced_accuracy_score(
                        y_true=y_category, y_pred=y_pred)
                    output_line = f"{category} {value} {balaced_accuracy}\n"
                    file.write(output_line)

    except ValueError:
        with open(args.data_slice_report, "a") as file:
            file.write("No instances for {}".format(value))


def save_pipeline(pipeline, path):
    """
    Saves trained pipeline in a local path

    Args:
        pipeline (Sklearn pipeline): Trained pipeline
        path (string): Local path to save the pipeline
    """
    joblib.dump(pipeline, path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Sklearn Pipeline training and evaluation")

    parser.add_argument(
        "--clean_data",
        type=str,
        help="Path to CSV File where cleaned data is stored"
    )

    parser.add_argument(
        "--cat_features",
        type=str,
        help="List of categorical columns to use for training"
    )

    parser.add_argument(
        "--target",
        type=str,
        help="Target variable"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Size (percentage in decimal) to split the training dataset",
        required=True
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Random state number for reproducibility",
        required=True
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        help="Ranfom forest hyperparameter",
        required=True
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        help="Random forest hyperparameter",
        required=True
    )

    parser.add_argument(
        "--max_features",
        type=str,
        help="Random forest hyperparameters",
        required=True
    )

    parser.add_argument(
        "--random_state_rf",
        type=int,
        help="Random state number for reproducibility for the Random Forest",
        required=True
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Enables using all the CPU cores for traing random forest",
        required=True
    )

    parser.add_argument(
        "--pipeline_path",
        type=str,
        help="Path to save trained pipeline",
        required=True
    )

    parser.add_argument(
        "--classification_report",
        type=str,
        help="Path to save trained pipeline",
        required=True
    )

    parser.add_argument(
        "--data_slice_report",
        type=str,
        help="Path to save data slice evaluation report",
        required=True
    )

    args = parser.parse_args()

    go(args)
