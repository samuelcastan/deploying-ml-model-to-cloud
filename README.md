## Install requeriments
```
pip install -r requierements
```

## HOW TO RUN AND TRAIN THE ENTIRE PIPELINE
```
mlflow run .
```

- Nothing should be changed unless you're going to modify how the data is splited, processed or even a new algorithm architecture
- To experiment with different set of hyperparmeters or so please change the values in the config.yaml file

## Consideration for a better development and enhancements to this project:
- Undersampling majority class might be doing more harm than benefit to the model performance. It is pending to do more experimentation and see if it is really benefitting performance
- It might be better to calibrate the model probabilities insted of the previous approach. E.g. use conformal prediction
- Save artifacts (raw data, cleaned data, trained pipeline, etc.) in a remote and versioned model registry. For simplicity of the project this was not required.