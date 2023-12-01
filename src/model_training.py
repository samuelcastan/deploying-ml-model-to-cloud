import pandas as pd
import joblib
import constants

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def balance_dataset(X, y, test_size, random_state):
    ''''
    Receives an imbalanced dataset and returns features and target variable with same proportions downsampling the mayority class

    Inputs:
        X: Independent variables
        y: Target or dependendent variable
    '''

    # First split to obtain testing dastaset and temporal set for balancing the classes
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Start the imbalanced dataset
    df_temp = pd.concat([X_temp, y_temp], axis=1)

    del X_temp, y_temp

    undersampling_size = len(df_temp[df_temp[constants.TARGET] == ">50K"])

    df_temp = pd.concat(
        [df_temp[df_temp[constants.TARGET] == "<=50K"].sample(undersampling_size),
         df_temp[df_temp[constants.TARGET] == ">50K"]]
    )

    X_train = df_temp[constants.CAT_FEATURES + constants.NUM_FEATURES]
    y_train = df_temp[constants.TARGET]

    return X_train, X_test, y_train, y_test


def make_pipeline(X_train, y_train):
    '''
    Trains the the entire ML inference pipeline: should train on the provided data. 

    Inputs:
        data: Dataframe containing features and target variables
        cat_features: List of categorical features 
        num_features: List of numerical features
        target: Target Variables

    Output:
        inference_pipeline: Pipeline that one-hot encodes categorical features and trains a Random Forest Model for later saving
    '''

    
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
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ('one-hot-encoder', column_transformer),
            ('classifier', classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline


if __name__ == '__main__':
    # Read data
    df = pd.read_csv(constants.TRAINING_DATASET)

    # Features and target variables
    X = df[constants.CAT_FEATURES + constants.NUM_FEATURES]
    y = df[constants.TARGET]

    X_train, X_test, y_train, y_test = balance_dataset(
        X, y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)
    
    pipeline = make_pipeline(X_train=X_train, y_train=y_train)

    # Save pipeline
    joblib.dump(pipeline, "model/inference_pipeline.pkl")