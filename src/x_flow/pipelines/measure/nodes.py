"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3
"""

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, TypeAlias, Union

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd
from joblib import Parallel, delayed
from x_flow.utils.data import PredictionData, ValidationPredictionData
from x_flow.utils.dr_helpers import get_training_predictions

from x_flow.utils.metrics.dr_metrics import DataRobotMetrics
from x_flow.utils.metrics.generalized_f1 import GeneralizedF1
from x_flow.utils.metrics.mean_squared_error import MeanSquaredError
from x_flow.utils.metrics.metrics import MetricFactory

log = logging.getLogger(__name__)


# # TODO:
# # min CV/BT thing (secondary_validation..) as default global setting


# def select_model_candidates(projects: List[dr.Project]) -> List[dr.Model]:
#     # select best model
#     ...


# def calculate_custom_metric_1(
#     actuals: pd.Series, predictions: pd.Series, other_data: Optional[pd.DataFrame]
# ) -> float:
#     # calculate custom metric
#     ...


# def read_project(projects: List[dr.Project]):
#     # select models
#     models = ...
#     # request and collect training predictions
#     # request and collect validation predictions
#     # request and collect holdout predictions

#     # calculate custom metric for each model, for each training and validation and holdout and external holdout
#     custom_metrics: dict[(partition, model), List[custom_metrics]] = ...


# # output:
# # list of best model per attribute_group


# def decay_test(model: dr.Model, decay_testing_params: dict) -> custom_metric_per_period:
#     #  decay testing to figure out how frequently to retrain
#     ...


MetricFactory.register_metric("generalized_f1", GeneralizedF1())
MetricFactory.register_metric("mean_squared_error", MeanSquaredError())
MetricFactory.register_metric("datarobot_metrics", DataRobotMetrics())

Metric: TypeAlias = Callable[
    [pd.Series, pd.Series, Optional[pd.DataFrame], Optional[dict], Optional[dict]],
    float,
]


def _load_and_index(row: pd.Series) -> pd.DataFrame:
    # Load the data from the function stored in the row
    try:
        load_function: Callable[[], ValidationPredictionData] = row["load_function"]
        validation_predictions = load_function()
        rendered_df = validation_predictions.rendered_df
        loaded_df = rendered_df
    except Exception as e:
        log.error(f"Error loading data: {e}")
        log.error(row["load_function"]().rendered_df)
        raise e

    # Assumption: this function returns a DataFrame
    # Create a DataFrame with repeated rows of index data for each entry in the loaded DataFrame
    index_data = pd.DataFrame(
        [row.drop("load_function")] * len(loaded_df),
        columns=row.index.drop("load_function"),
    )

    # Concatenate the index data with the loaded DataFrame
    combined_df = pd.concat(
        [index_data.reset_index(drop=True), loaded_df.reset_index(drop=True)], axis=1
    )

    return combined_df


def calculate_metrics(
    experiment_config: dict,
    predictions: dict[str, ValidationPredictionData],
    metric_config: dict,
    metrics: List[str],
):
    experiment_name = experiment_config["experiment_name"]
    metadata_df = pd.DataFrame(
        [k.replace(".csv", "").split("/")[:-1] for k in predictions]
    )
    metadata_df.columns = [
        "partition",
        "project_id",
        "model_id",
        "data_subset_name",
    ]
    metadata_df["load_function"] = predictions.values()

    def process_group(
        data_subset: dr.enums.DATA_SUBSET,
        df: pd.DataFrame,
        experiment_name: str,
        metrics: List[str],
        experiment_config: dict,
        metric_config: dict,
    ):
        # This will store all the metadata for the current group
        group_metrics_list = []

        for (project_id, model_id), model_df in df.groupby(["project_id", "model_id"]):
            load_functions = model_df["load_function"].to_list()
            validation_predictions = [
                load_function() for load_function in load_functions
            ]
            all_predictions = []
            for validation_prediction in validation_predictions:
                rendered_df = validation_prediction.rendered_df
                rendered_df["project_id"], rendered_df["model_id"] = (
                    project_id,
                    model_id,
                )
                all_predictions.append(rendered_df)
            all_predictions = pd.concat(all_predictions)

            target_column = {
                validation_prediction.target_column
                for validation_prediction in validation_predictions
            }

            if len(target_column) > 1:
                raise ValueError(f"Multiple target columns found: {target_column}")

            target_column = target_column.pop()

            metadata = {
                "experiment_name": experiment_name,
                "project_id": project_id,
                "data_subset": data_subset,
                "model_id": model_id,
            }
            for metric in metrics:
                metric_instance = MetricFactory.get_metric(metric)
                metric_value = metric_instance.compute(
                    actuals=all_predictions[target_column],
                    predictions=all_predictions["prediction"],
                    experiment_config=experiment_config,
                    metric_config=metric_config,
                    extra_data=all_predictions,
                    metadata=metadata,
                )

                metadata.update(metric_value)
            group_metrics_list.append(metadata)
            # log.info(f"metrics: {metadata}")

        return group_metrics_list

    tasks = (
        delayed(process_group)(
            data_subset,
            df,
            experiment_name,
            metrics,
            experiment_config,
            metric_config,
        )
        for data_subset, df in metadata_df.groupby("data_subset_name")
    )

    results = Parallel(n_jobs=10)(tasks)

    metrics_list = [item for sublist in results for item in sublist]
    return pd.DataFrame(metrics_list)
