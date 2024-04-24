"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3
"""

from typing import List, Optional

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd


# TODO:
# min CV/BT thing (secondary_validation..) as default global setting


def select_model_candidates(projects: List[dr.Project]) -> List[dr.Model]:
    # select best model
    ...


def calculate_custom_metric_1(
    actuals: pd.Series, predictions: pd.Series, other_data: Optional[pd.DataFrame]
) -> float:
    # calculate custom metric
    ...


def read_project(projects: List[dr.Project]):
    # select models
    models = ...
    # request and collect training predictions
    # request and collect validation predictions
    # request and collect holdout predictions

    # calculate custom metric for each model, for each training and validation and holdout and external holdout
    custom_metrics: dict[(partition, model), List[custom_metrics]] = ...


# output:
# list of best model per attribute_group


def decay_test(model: dr.Model, decay_testing_params: dict) -> custom_metric_per_period:
    #  decay testing to figure out how frequently to retrain
    ...
