import os
import hydra
import tempfile
import mlflow
from omegaconf import DictConfig

_steps = [
    "basic_cleaning"
    # "data_check",
    # "data_split",
    # "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before
    # you can run this, then you need to run this step explicitly
    #  "test_regression_model"
]


@hydra.main(version_base=None, config_path='config', config_name="config")
def go(config: DictConfig) -> None:

    # Steps to execute
    # steps_par = config['main']['steps']
    # active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "basic_cleaning" in _steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "basic_cleaning"),
                entry_point="main",
                parameters={
                    "raw_data_path": config["file_paths"]["raw_data"],
                    "clean_data_path": config["file_paths"]["clean_data"],
                    "keep_columns": config["data"]["keep_columns"]},
            )

        if "train_pipeline" in _steps:
            pass
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "train_pipeline"
                ),
                entry_point="main",
                parameters={
                    "clean_data": config["file_paths"]["clean_data"],
                    "cat_features": config["data"]["categorical_features"],
                    "target": config["data"]["target"],
                    "test_size": config["training_hyperparameters"]["test_size"],
                    "random_state": config["random_state"],
                    "n_estimators": config["training_hyperparameters"]["random_forest"]["n_estimators"],
                    "max_depth": config["training_hyperparameters"]["random_forest"]["max_depth"],
                    "max_features": config["training_hyperparameters"]["random_forest"]["max_features"],
                    "random_state_rf": config["training_hyperparameters"]["random_forest"]["random_state_rf"],
                    "n_jobs": config["training_hyperparameters"]["random_forest"]["n_jobs"],
                    "pipeline_path": config["file_paths"]["pipeline"],
                    "classification_report": config["file_paths"]["classification_report"],
                    "data_slice_report": config["file_paths"]["data_slice_report"]
                }
            )


if __name__ == "__main__":

    go()
