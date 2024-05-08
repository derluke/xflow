"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3
"""

import logging
from functools import partial
from typing import Any, Callable, List, Optional, TypeAlias, Union

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd
from utils.dr_helpers import get_training_predictions

from x_flow.metrics.implementations import GeneralizedF1, MeanSquaredError
from x_flow.metrics.metrics import MetricFactory

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

Metric: TypeAlias = Callable[
    [pd.Series, pd.Series, Optional[pd.DataFrame], Optional[dict], Optional[dict]],
    float,
]


def _load_and_index(row: pd.Series, columns: Optional[List[str]] = None):

    # Load the data from the function stored in the row
    try:
        loaded_df = row["load_function"]()[columns]
    except Exception as e:
        log.error(f"Error loading data: {e}")
        log.error(row["load_function"]().columns)
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


def calculate_custom_metric(
    actuals: pd.Series,
    predictions: pd.Series,
    metric_name: str,
    experiment_config: dict,
    metric_config: Optional[dict] = None,
    extra_data: Optional[pd.DataFrame] = None,
) -> float:
    metric = MetricFactory.get_metric(metric_name)
    return metric.compute(
        actuals, predictions, extra_data, experiment_config, metric_config
    )


def calculate_metrics(
    experiment_name: str,
    experiment_config: dict,
    predictions: dict[str, pd.DataFrame],
    metric_config: dict,
    target_binarized: str,
    metrics: List[str],
):

    metadata_df = pd.DataFrame([k.split("/") for k in predictions])
    metadata_df.columns = [
        "partition",
        "project_id",
        "model_id",
        "data_subset_name",
    ]
    metadata_df["load_function"] = predictions.values()

    all_subgroups = pd.concat(
        metadata_df.apply(partial(_load_and_index, columns=["prediction", target_binarized]), axis=1).tolist(), ignore_index=True  # type: ignore
    )
    metrics_list = []
    for group, df in all_subgroups.groupby("data_subset_name"):
        for (project_id, model_id), model_df in df.groupby(["project_id", "model_id"]):
            row = {
                "experiment_name": experiment_name,
                "project_id": project_id,
                "group": group,
                "model_id": model_id,
            }
            for metric in metrics:
                metric_value = calculate_custom_metric(
                    actuals=model_df[target_binarized],
                    predictions=model_df["prediction"],
                    metric_name=metric,
                    experiment_config=experiment_config,
                    metric_config=metric_config,
                )

                row[metric] = metric_value
            metrics_list.append(row)

    return pd.DataFrame(metrics_list)
    # actuals = get_actuals(v, project_dict["target"])
    # for metric in metrics:
    #     metric_value = calculate_custom_metric(
    #         actuals,
    #         v["prediction"],
    #         metric,
    #         project_dict["experiment_config"],
    #         other_data=None,
    #         metric_config=None,
    #     )
    #     print(
    #         f"experiment_name: {experiment_name}, model_id: {project_dict['model_id']}, metric: {metric.__name__}, metric_value: {metric_value}"
    #     )


# def select_candidate_models(
#     projects: List[dict[str, Any]],
#     metric: Metric = generalized_f1,
#     metric_config: Optional[dict] = None,
#     data_subset: dr.enums.DATA_SUBSET = dr.enums.DATA_SUBSET.HOLDOUT,
# ) -> List[dr.Model]:
#     # select best model
#     candidate_models = []
#     for row in projects:
#         experiment_name = row["experiment_name"]
#         project = dr.Project.get(row["project_id"])
#         experiment_config = row.get("experiment_config", {})
#         models = project.get_models()
#         project_metrics = []
#         for model in models:
#             training_predictions = get_training_predictions(model, data_subset)
#             actuals = training_predictions[
#                 experiment_config["analyze_and_model"]["target"]
#             ]
#             predictions = training_predictions["prediction"]
#             metric_value = metric(
#                 actuals, predictions, None, metric_config, experiment_config
#             )
#             project_metrics.append(
#                 {
#                     "experiment_name": experiment_name,
#                     "model_id": model.id,
#                     "project_id": project.id,
#                     "metric_value": metric_value,
#                 }
#             )
#         best_models = sorted(
#             project_metrics, key=lambda x: x["metric_value"], reverse=True
#         )[:5]
#         candidate_models.extend(best_models)
#     return candidate_models
