# Model Card: Random Forest Classifier

## Overview

**Model Name:** Random Forest Classifier

**Model Version:** 1.0

**Date of Last Update:** January 2024

**Author:** Samuel David Castán Alejandre

## Model Information

**Purpose:** The Random Forest Classifier is designed for classifying the salary (lower than 50k or greater or equal than 50k yearly) of a person in the U.S.

**Intended Use:** The model is intended to assist [target users] in making predictions for [specific use cases].

**Inputs:** The model takes categorical inputs such as age, profession, education, etc. as input.

**Outputs:** The model provides two labels; ***<50k*** or ***>=50K*** correspondent to how much salary someone earns

## Model Architecture

**Algorithm:** Random Forest

**Number of Trees:** 100

**Other Hyperparameters:**
- n_jobs=-1
- max_depth=15
- max_features="sqrt"
- random_state=42

## Consideration for future improvements:
- Undersampling majority class might be doing more harm than benefit to the model performance. It is pending to do more experimentation and see if it is really benefitting performance
- It might be better to calibrate the model probabilities insted of the previous approach. E.g. use conformal prediction
- Save artifacts in a remote and versioned model registry (e.g. raw data, cleaned data, trained pipeline, etc.)

## Limitations
- Data used from https://archive.ics.uci.edu/dataset/20/census+income which was last updated/created on 1994. Surely this dataset doesn't resemble the current reality (2024).

## Performance Metrics on Testing Dataset
- Recall (>50K): 0.8634
- Recall (<=50K): 0.6866
- Precision (>50K): 0.4738
- Precision (<=50K): 0.9389

## Data Slice Evaluation
- Refer to *model/data_slice_report.txt* to see thoroughly the slice evaluation to check for possible biases.