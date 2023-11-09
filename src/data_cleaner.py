''''
Script that removes whitespaces on the column names

Author: Samuel Castan
Created: Oct 2023
Last Modified: Oct 2023
'''

import pandas as pd

df = pd.read_csv("data/raw/census.csv")

df.columns = [column.replace(" ", "") for column in df.columns]

columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "salary"]

for column in columns:
    df[column] = df[column].str.replace(" ", "")

# Remove duplicates
df = df.drop_duplicates()

# Export cleaned dataset
df.to_csv("data/clean/census.csv", index=False)