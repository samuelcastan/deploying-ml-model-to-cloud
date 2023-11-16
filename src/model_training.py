import pandas as pd
import globals

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Read data
df = pd.read_csv('data/clean/census.csv')

# Features and target variables
X = df[globals.CAT_FEATURES + globals.NUM_FEATURES]
y = df[globals.TARGET]

# First split to obtain testing dastaset and temporal set for balancing the classes
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_temp = pd.concat([X_temp, y_temp], axis=1)

del X_temp, y_temp

undersampling_size = len(df_temp[df_temp["salary"]==">50K"])

df_temp = pd.concat(
    [df_temp[df_temp["salary"] == "<=50K"].sample(undersampling_size), 
     df_temp[df_temp["salary"] == ">50K"]]
)

X_train = df_temp[globals.CAT_FEATURES + globals.NUM_FEATURES]
y_train = df_temp[globals.TARGET]