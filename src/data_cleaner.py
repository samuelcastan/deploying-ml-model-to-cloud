''''
Script that removes whitespaces on the column names

Author: Samuel Castan
Created: Oct 2023
Last Modified: Oct 2023
'''
import pandas as pd


def clean_data():
    
    # Read raw data
    df = pd.read_csv("data/raw/census.csv")

    # Remove whitespaces to column names
    df.columns = [column.replace(" ", "") for column in df.columns]

    # Columns to remove also whitespaces
    columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "salary"]

    # Remove whitespaces to the values of categorical columns plus target variables
    for column in columns:
        df[column] = df[column].str.replace(" ", "")

    # Remove duplicates
    df = df.drop_duplicates()

    # Export cleaned dataset
    df.to_csv("data/clean/census.csv", index=False)


if __name__ == '__main__':
    clean_data()