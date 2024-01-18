''''
Script that removes whitespaces on the column names and values in
categorical columns plus target variable

Author: Samuel Castan
Created: Oct 2023
Last Modified: Oct 2023
'''

import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    df = pd.read_csv(args.raw_data_path)
    logger.info("INFO: RAW DATA READ SUCCESFULLY")

    df.columns = [column.replace(" ", "") for column in df.columns]
    df.columns = [column.replace("-", "_") for column in df.columns]
    logger.info("INFO: WHITESPACES REMOVED IN COLUMN NAMES")

    columns = args.keep_columns.split(",")  # Transform string to list

    for column in columns:
        df[column] = df[column].str.replace(" ", "")
    logger.info("INFO: WHITESPACES REMOVED IN COLUMN VALUES")

    df = df.drop_duplicates()
    logger.info("INFO: DUPLICATED ROWS REMOVED")

    df.to_csv(args.clean_data_path, index=False)
    logger.info("INFO: SAVED CLEANED DATA SUCCESFULLY")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--raw_data_path",
        type=str,
        help="Path to the raw data to clean",
        required=True
    )

    parser.add_argument(
        "--clean_data_path",
        type=str,
        help="Path to the clean data to save",
        required=True
    )

    parser.add_argument(
        "--keep_columns",
        type=str,
        help="Columns to keep for the cleaned dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
