"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TypeAlias

from joblib import Parallel, delayed
import pandas as pd

from x_flow.utils.data import ValidationPredictionData
from x_flow.utils.metrics.dr_metrics import DataRobotMetrics
from x_flow.utils.metrics.generalized_f1 import GeneralizedF1
from x_flow.utils.metrics.mean_squared_error import MeanSquaredError
from x_flow.utils.metrics.metrics import MetricFactory

# pyright: reportPrivateImportUsage=false
import datarobot as dr

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


def _get_predictions_and_target(model_df):
    load_functions = model_df["load_function"].to_list()
    validation_predictions = [load_function() for load_function in load_functions]
    all_predictions_list = []
    for validation_prediction in validation_predictions:
        rendered_df = validation_prediction.rendered_df
        all_predictions_list.append(rendered_df)
    all_predictions = pd.concat(all_predictions_list)

    target_column = {
        validation_prediction.target_column for validation_prediction in validation_predictions
    }
    if len(target_column) > 1:
        raise ValueError(f"Multiple target columns found: {target_column}")
    target_column = target_column.pop()
    return all_predictions, target_column


def calculate_metrics(
    experiment_config: Dict[str, Any],
    prediction_data: Dict[str, ValidationPredictionData],
    metric_config: Dict[str, Any],
    metrics: List[str],
    best_models: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    experiment_name = experiment_config["experiment_name"]
    metadata_df = _prepare_metadata(prediction_data, experiment_name, best_models)
    group_by_partition = best_models is None or best_models.empty

    tasks = [
        delayed(_process_group)(
            data_subset,
            df,
            experiment_name,
            metrics,
            experiment_config,
            metric_config,
            group_by_partition,
        )
        for data_subset, df in metadata_df.groupby("data_subset_name")
    ]

    results = Parallel(n_jobs=10)(tasks)
    metrics_list = [item for sublist in results for item in sublist]
    return pd.DataFrame(metrics_list)


def _prepare_metadata(
    prediction_data: Dict[str, ValidationPredictionData],
    experiment_name: str,
    best_models: Optional[pd.DataFrame],
) -> pd.DataFrame:
    metadata_df = pd.DataFrame([k.replace(".csv", "").split("/")[:-1] for k in prediction_data])
    metadata_df.columns = ["partition", "project_id", "model_id", "data_subset_name"]  # type: ignore
    metadata_df["load_function"] = prediction_data.values()

    if best_models is not None and not best_models.empty:
        best_models = best_models[best_models["experiment_name"] == experiment_name]
        metadata_df = metadata_df.merge(best_models, on=["project_id", "model_id"], how="inner")

    return metadata_df


def _process_group(
    data_subset: dr.enums.DATA_SUBSET,
    df: pd.DataFrame,
    experiment_name: str,
    metrics: List[str],
    experiment_config: Dict[str, Any],
    metric_config: Dict[str, Any],
    group_by_partition: bool = True,
) -> List[Dict[str, Any]]:
    groupby_columns = ["partition", "project_id", "model_id"] if group_by_partition else ["rank"]

    return [
        _calculate_group_metrics(
            group_key,
            model_df,
            experiment_name,
            data_subset,
            metrics,
            experiment_config,
            metric_config,
            group_by_partition,
        )
        for group_key, model_df in df.groupby(groupby_columns)
    ]


def _calculate_group_metrics(
    group_key: Any,
    model_df: pd.DataFrame,
    experiment_name: str,
    data_subset: dr.enums.DATA_SUBSET,
    metrics: List[str],
    experiment_config: Dict[str, Any],
    metric_config: Dict[str, Any],
    group_by_partition: bool,
) -> Dict[str, Any]:
    metadata = _get_group_metadata(
        group_key, model_df, experiment_name, data_subset, group_by_partition
    )
    all_predictions, target_column = _get_predictions_and_target(model_df)

    if metadata["model_id"] is not None and "," not in metadata["model_id"]:
        model_id = metadata["model_id"]
        all_predictions["model_id"] = model_id

    if metadata["project_id"] is not None and "," not in metadata["project_id"]:
        project_id = metadata["project_id"]
        all_predictions["project_id"] = project_id

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

    return metadata


def _get_group_metadata(
    group_key: Any,
    model_df: pd.DataFrame,
    experiment_name: str,
    data_subset: dr.enums.DATA_SUBSET,
    group_by_partition: bool,
) -> Dict[str, Any]:
    if group_by_partition:
        partition, project_id, model_id = group_key
        rank = None
    else:
        rank = group_key[0]
        partition = None
        project_id = (
            model_df["project_id"].iloc[0] if model_df["project_id"].nunique() == 1 else None
        )
        model_id = model_df["model_id"].iloc[0] if model_df["model_id"].nunique() == 1 else None

    metadata = {
        "experiment_name": experiment_name,
        "partition": partition,
        "project_id": project_id or ",".join(map(str, model_df["project_id"].unique())),
        "data_subset": data_subset,
        "model_id": model_id or ",".join(map(str, model_df["model_id"].unique())),
        "rank": rank,
    }

    return metadata


def get_best_models(
    metrics_by_partition: pd.DataFrame,
    experiment_config: Dict[str, Any],
) -> pd.DataFrame:
    main_metric = experiment_config["main_metric"]
    group_by_partition = len(metrics_by_partition["partition"].unique()) > 1

    if group_by_partition:
        grouped = metrics_by_partition.groupby(["experiment_name", "partition", "project_id"])
        best_models = grouped.apply(lambda x: x.loc[x[main_metric].idxmax()]).reset_index(drop=True)  # type: ignore
        best_models["rank"] = "1"
    else:
        best_models = metrics_by_partition.sort_values(by=main_metric, ascending=False)[:1]
        best_models["rank"] = range(1, len(best_models) + 1)

    return best_models[["experiment_name", "rank", "project_id", "model_id", main_metric]]
