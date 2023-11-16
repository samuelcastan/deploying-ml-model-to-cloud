import pandas as pd
import joblib
import globals

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Read data
df = pd.read_csv('data/clean/census.csv')

# Features and target variables
X = df[globals.CAT_FEATURES + globals.NUM_FEATURES]
y = df[globals.TARGET]

# First split to obtain testing dastaset and temporal set for balancing the classes
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Start the imbalanced dataset
df_temp = pd.concat([X_temp, y_temp], axis=1)

del X_temp, y_temp

undersampling_size = len(df_temp[df_temp["salary"]==">50K"])

df_temp = pd.concat(
    [df_temp[df_temp["salary"] == "<=50K"].sample(undersampling_size), 
     df_temp[df_temp["salary"] == ">50K"]]
)

X_train = df_temp[globals.CAT_FEATURES + globals.NUM_FEATURES]
y_train = df_temp[globals.TARGET]



column_transformer = ColumnTransformer(
    transformers=[
        # name, transformer, columns
        ('cat', OneHotEncoder(handle_unknown="ignore"), globals.CAT_FEATURES),
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


# Save model
joblib.dump(pipeline, "model/inference_pipeline.pkl")