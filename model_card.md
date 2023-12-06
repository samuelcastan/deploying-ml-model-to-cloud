# Model Card: Random Forest Classifier

## Overview

**Model Name:** Random Forest Classifier

**Model Version:** 1.0

**Date of Last Update:** November 2023

**Author:** Samuel David Castán Alejandre

## Model Information

**Purpose:** The Random Forest Classifier is designed for classifying the salary (lower than 50k or greater or equal than 50k yearly) of a person in the U.S.

**Intended Use:** The model is intended to assist [target users] in making predictions for [specific use cases].

**Inputs:** The model takes numerical and categorical inputs such as age, profession, education, etc. as input.

**Outputs:** The model provides two labels; "<50k salary" or ">=50K salary"

## Model Architecture

**Algorithm:** Random Forest

**Number of Trees:** 100

**Other Hyperparameters:** 
- n_jobs=-1,
- max_depth=15,
- max_features="sqrt",
- random_state=42

## Performance

### Training Data

- **Training Dataset Size:** [Number of instances in the training set]
- **Training Time:** [Time taken for training]
- **Training Metrics:** [List metri
