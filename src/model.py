import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score
import constants

warnings.filterwarnings("ignore")


def balance_dataset(X, y, test_size, random_state):
    """"
    Receives an imbalanced dataset and returns features and target variable with the same proportions downsampling the majority class

    Inputs:
        X: Independent variables
        y: Target or dependent variable
        test_size: Percentage of the dataset to be tested on
        random_state: Reproducibility number
    """

    # First split to obtain the testing dataset and temporal set for balancing
    # the classes
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Start the imbalanced dataset
    df_temp = pd.concat([X_temp, y_temp], axis=1)

    del X_temp, y_temp

    undersampling_size = len(df_temp[df_temp[constants.TARGET] == ">50K"])

    df_temp = pd.concat([df_temp[df_temp[constants.TARGET] == "<=50K"].sample(
        undersampling_size), df_temp[df_temp[constants.TARGET] == ">50K"]])

    X_train = df_temp[constants.CAT_FEATURES + constants.NUM_FEATURES]
    y_train = df_temp[constants.TARGET]

    return X_train, X_test, y_train, y_test


def train_pipeline(X_train, y_train, random_state=42):
    """""
    Trains the entire ML inference pipeline: should train on the provided data.

    Inputs:
        X_train: Instances with features to train on
        y_train: Target variable

    Output:
        pipeline: Pipeline that one-hot encodes categorical features and trains a Random Forest Model for later saving
    """

    column_transformer = ColumnTransformer(
        transformers=[
            # name, transformer, columns
            ('cat', OneHotEncoder(handle_unknown="ignore"), constants.CAT_FEATURES),
        ],
        # Ignore numerical columns
        remainder='passthrough'
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        max_depth=15,
        max_features="sqrt",
        random_state=random_state
    )

    pipeline = Pipeline(
        steps=[
            ('one-hot-encoder', column_transformer),
            ('classifier', classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    model_performance(pipeline, X_test=X_test, y_test=y_test)

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
    file_path = './model/classification_report.txt'

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
        cat_features (list, optional): cateogorical features to slice on. Defaults to ["education"].
    """

    df_temp = pd.concat([X_test, y_test], axis=1)

    try:
        with open("model/data_slice_performance.txt", "w") as file:
            file.write("Category - Value - Balanced Acccuracy\n")
            for category in cat_features:
                for value in df[category].unique():
                    X_category = df_temp[df_temp[category]
                                         == value].drop(["salary"], axis=1)
                    y_category = df_temp[df_temp[category] == value]["salary"]

                    y_pred = pipeline.predict(X_category)

                    balaced_accuracy = balanced_accuracy_score(
                        y_true=y_category, y_pred=y_pred)
                    output_line = f"{category} {value} {balaced_accuracy}\n"
                    file.write(output_line)

    except ValueError:
        with open("model/data_slice_performance.txt", "a") as file:
            file.write("No instances for {}".format(value))


def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)


if __name__ == '__main__':
    # Read data
    df = pd.read_csv(constants.TRAINING_DATASET)

    # Features and target variables
    X = df[constants.CAT_FEATURES + constants.NUM_FEATURES]
    y = df[constants.TARGET]

    X_train, X_test, y_train, y_test = balance_dataset(
        X, y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)

    # train pipeline
    pipeline = train_pipeline(
        X_train=X_train,
        y_train=y_train,
        random_state=constants.RANDOM_STATE)

    # evaluate model
    model_performance(pipeline=pipeline, X_test=X_test, y_test=y_test)

    data_slicing_evaluation(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        cat_features=constants.CAT_FEATURES)

    # Save pipeline
    save_pipeline(pipeline=pipeline, path=constants.PIPELINE_PATH)
