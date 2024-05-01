"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3
"""

from typing import Any, Callable, List, Optional, TypeAlias, Union

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd

from x_flow.utils.dr_helpers import get_training_predictions


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

Metric: TypeAlias = Callable[
    [pd.Series, pd.Series, Optional[pd.DataFrame], Optional[dict], Optional[dict]],
    float,
]

from ...metrics.generalized_auc import generalized_f1





# def get_training_predictions(
#     model: dr.Model, data_subset: dr.enums.DATA_SUBSET
# ) -> pd.DataFrame:
#     try:
#         pred_job = model.request_training_predictions(data_subset=data_subset)
#         tp = pred_job.get_result_when_complete()
#     except Exception:  # pylint: disable=broad-except
#         all_training_predictions = dr.TrainingPredictions.list(
#             project_id=model.project_id
#         )
#         tp = [
#             tp
#             for tp in all_training_predictions
#             if tp.model_id == model.id and tp.data_subset == data_subset
#         ][0]
#     return tp.get_all_as_dataframe()  # type: ignore




def get_actuals(df: pd.DataFrame, target: str):
    return df[target]


def calculate_custom_metric(
    actuals: pd.Series,
    predictions: pd.Series,
    metric: Metric,
    experiment_config: dict,
    other_data: Optional[pd.DataFrame] = None,
    metric_config: Optional[dict] = None,
) -> float:
    return metric(actuals, predictions, other_data, metric_config, experiment_config)


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
