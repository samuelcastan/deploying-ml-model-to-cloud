## To know more about model development please refer to the [model card](model_card.md) file

## Install requeriments
```
pip install -r requierements
```

## TRAIN THE ENTIRE PIPELINE
```
mlflow run .
```
## RUN A SPECIFIC TASK
```
mlflow run . -P steps=basic_cleaning
mlflow run . -P steps=train_pipeline
```
 ## TESTING
 Pytest was configured to run from the root directory following with the next command:
 ```
 pytest -vv
 ```


## Note
- Nothing should be changed unless you're going to modify how the data is splited, processed or even a new algorithm architecture
- To experiment with different set of hyperparmeters or so please change the values in the config.yaml file


