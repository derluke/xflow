"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3.
"""

import logging
from typing import Any, Callable, List, Optional, TypeAlias

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
    experiment_config: dict[str, Any],
    predictions: dict[str, ValidationPredictionData],
    metric_config: dict[str, Any],
    metrics: List[str],
    best_models: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    experiment_name = experiment_config["experiment_name"]
    metadata_df = pd.DataFrame([k.replace(".csv", "").split("/")[:-1] for k in predictions])
    metadata_df.columns = [  # type: ignore[assignment]
        "partition",
        "project_id",
        "model_id",
        "data_subset_name",
    ]
    metadata_df["load_function"] = predictions.values()
    group_by_partition = True
    if best_models is not None:
        best_models = best_models[best_models["experiment"] == experiment_name]
        if best_models is None or best_models.shape[0] == 0:
            log.warning(f"No best models found for experiment {experiment_name}")
            group_by_partition = True
        else:
            metadata_df = (
                metadata_df.set_index(["project_id", "model_id"])
                .join(
                    best_models.set_index(["project_id", "model_id"]),
                    on=["project_id", "model_id"],
                    how="inner",
                )
                .reset_index()
            )
            group_by_partition = False

    def process_group(
        data_subset: dr.enums.DATA_SUBSET,
        df: pd.DataFrame,
        experiment_name: str,
        metrics: list[str],
        experiment_config: dict[str, Any],
        metric_config: dict[str, Any],
        group_by_partition: bool = True,
    ) -> list[dict[str, Any]]:
        # This will store all the metadata for the current group
        group_metrics_list = []

        # Define groupby columns based on the group_by_partition flag
        groupby_columns = (
            ["partition", "project_id", "model_id"] if group_by_partition else ["rank"]
        )

        for group_key, model_df in df.groupby(groupby_columns):
            # Unpack group_key based on groupby_columns
            if group_by_partition:
                partition, project_id, model_id = group_key
                rank = None
            else:
                (rank,) = group_key
                partition = None
                if model_df["project_id"].nunique() > 1:
                    model_id = None
                else:
                    model_id = model_df.iloc[0]["model_id"]
                if model_df["project_id"].nunique() > 1:
                    project_id = None
                else:
                    project_id = model_df.iloc[0]["project_id"]

            all_predictions, target_column = _get_predictions_and_target(model_df)

            all_predictions["project_id"], all_predictions["model_id"] = project_id, model_id
            metadata = {
                "experiment_name": experiment_name,
                "partition": partition,
                "project_id": project_id,
                "data_subset": data_subset,
                "model_id": model_id,
                "rank": rank,
            }

            if metadata["project_id"] is None:
                metadata["project_id"] = ",".join([str(i) for i in model_df["project_id"].unique()])
            if metadata["model_id"] is None:
                metadata["model_id"] = ",".join([str(i) for i in model_df["model_id"].unique()])

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
            group_by_partition,
        )
        for data_subset, df in metadata_df.groupby("data_subset_name")
    )

    results = Parallel(n_jobs=10)(tasks)

    metrics_list = [item for sublist in results for item in sublist]  # type: ignore
    metrics_by_partition = pd.DataFrame(metrics_list)

    return metrics_by_partition


def get_best_models(
    metrics_by_partition: pd.DataFrame,
    experiment_config: dict[str, Any],
) -> pd.DataFrame:
    best_models: list[dict[str, str]] = []
    main_metric = experiment_config["main_metric"]
    group_by_partition = len(metrics_by_partition["partition"].unique()) > 1
    # Determine grouping columns
    if group_by_partition:
        # Group by experiment, partition, and project_id
        grouped = metrics_by_partition.groupby(["experiment_name", "partition", "project_id"])

        for (experiment, partition, project_id), group in grouped:
            # Find the best model for this group
            best_model = group.loc[group[main_metric].idxmax()]

            best_models.append(
                {
                    "experiment": experiment,
                    "rank": "1",
                    # "partition": partition,
                    "project_id": project_id,
                    "model_id": str(best_model["model_id"]),
                    main_metric: str(best_model[main_metric]),
                }
            )
    else:
        # Group only by experiment
        grouped = metrics_by_partition.groupby("experiment_name")  # type: ignore

        for experiment, group in grouped:
            # Sort all models by the main metric in descending order
            sorted_models = group.sort_values(by=main_metric, ascending=False)
            for i, (_, row) in enumerate(sorted_models.iterrows()):
                best_models.append(
                    {
                        "experiment": experiment,
                        "rank": f"{i + 1}",
                        # "partition": "all",
                        "project_id": str(row["project_id"]),
                        "model_id": str(row["model_id"]),
                        main_metric: str(row[main_metric]),
                    }
                )

    return pd.DataFrame.from_records(best_models)
