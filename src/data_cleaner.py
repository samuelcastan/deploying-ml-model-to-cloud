''''
Script that removes whitespaces on the column names

Author: Samuel Castan
Created: Oct 2023
Last Modified: Oct 2023
'''

import pandas as pd

df = pd.read_csv("data/raw/census.csv")

df.columns = [column.replace(" ", "") for column in df.columns]

cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

for feature in cat_features:
    df[feature] = df[feature].str.replace(" ", "")

df.to_csv("data/clean/census.csv", index=False)