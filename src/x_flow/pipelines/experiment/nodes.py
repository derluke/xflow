"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.19.3
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd
from datarobot.rest import RESTClientObject
from datarobotx.idp.autopilot import get_or_create_autopilot_run
from datarobotx.idp.common.hashing import get_hash
from datarobotx.idp.datasets import get_or_create_dataset_from_df
from filelock import FileLock
from joblib import Parallel, delayed
from utils.dr_helpers import (
    RateLimiterSemaphore,
    get_external_holdout_predictions,
    get_models,
    get_training_predictions,
    wait_for_jobs,
)
from utils.operator import Operator

try:
    from datarobot import UseCase
except:

    class UseCase:
        def __init__(self, id: Optional[str], name: str):
            self.id = id
            self.name = name


# TODO:
# timeseries nowcasting support
# double threshold "binarization" support (multi project)
# low or other vs high or other.
# pre processing and post processing nodes (with factory and registry) (P1.5)
# custom feature selection


# monkey patch datarobot RESTClientObject to use a rate limiter
rate_limiter = RateLimiterSemaphore(30)

client_request_fun = RESTClientObject.request


def request_with_rate_limiter(*args, **kwargs):
    rate_limiter.acquire()
    response = client_request_fun(*args, **kwargs)
    rate_limiter.release()
    return response


RESTClientObject.request = request_with_rate_limiter

log = logging.getLogger(__name__)

dr_logger = logging.getLogger("datarobot.models.project")
for handler in dr_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        # log.info(f"Removing handler {handler}")
        dr_logger.removeHandler(handler)


def unpack_row_to_args(
    control_series: Dict, arg_look: Dict, arg_values_dict=None, verbose=False
) -> dict[str, Any]:
    if arg_values_dict is None:
        # initialise the dict with a control entry
        arg_values_dict = {"_control": {}}

    for param, param_value in control_series.items():
        if param in arg_look.keys():
            target_object = arg_look[param]

            # do nothing if the target object is ambiguous
            if target_object is not None:
                # init entry for target object if we don't already have
                if pd.notnull(param_value):
                    if target_object not in arg_values_dict.keys():
                        arg_values_dict[target_object] = {}
                    arg_values_dict[target_object][param] = param_value

                if verbose:
                    log.info(f"{param}: {param_value}, {target_object}")

        elif pd.notnull(param_value) and param[0] == "_":
            # non-null values for control parameters (prefixed with '_') are assigned to the control dict
            arg_values_dict["_control"][param] = param_value

        # TODO: handle ambiguous arguments better

    if verbose:
        log.info(f"-> {arg_values_dict}")

    return arg_values_dict


def binarize_data_if_specified(  # noqa: PLR0913
    input_data: pd.DataFrame,
    target: str,
    binarize_threshold=0.0,
    binarize_drop_regression_target=True,
    binarize_operator: Optional[str] = None,
    binarize_new_target_name="target_cat",
) -> tuple[pd.DataFrame, str]:
    """helper function: binarize a target variable for classification"""
    categorical_data = input_data.copy()
    if binarize_operator is None:
        return input_data, target
    op_fun = Operator(operator=binarize_operator).apply_operation(binarize_threshold)

    categorical_data[binarize_new_target_name] = categorical_data[target].apply(op_fun)
    if binarize_drop_regression_target:
        categorical_data.drop(columns=[target], inplace=True)

    return categorical_data, binarize_new_target_name


def binarize_data_node(
    input_data: pd.DataFrame, target: str, binarize_data_config: Dict
) -> tuple[pd.DataFrame, str]:

    return binarize_data_if_specified(
        input_data=input_data,
        target=target,
        **binarize_data_config if binarize_data_config else {},
    )


def get_or_create_dataset_from_df_with_lock(
    token: str,
    endpoint: str,
    use_case_id: str,
    df: pd.DataFrame,
    name: str,
    group_data: Optional[Dict] = None,
) -> Tuple[Dict[str, str], Dict[str, pd.DataFrame]]:
    # log.warning(f"Grouping data by {group_data['groupby_column']} for experiment {name}")

    if use_case_id == "not_supported":
        use_case_id = None

    def _get_or_create_dataset_from_df_with_lock(
        use_case_id: str,
        df: pd.DataFrame,
        name: str,
        group: Optional[str] = None,
    ) -> str:

        name = f"{name}_{group}" if (group != "__all_data__") else name
        df_token = get_hash(df, use_case_id, name)

        with FileLock(
            os.path.join(".locks", f"get_or_create_dataset_from_df_{df_token}.lock")
        ):
            df_id = get_or_create_dataset_from_df(
                token=token,
                endpoint=endpoint,
                use_cases=use_case_id,
                data_frame=group_df,
                name=name,
            )
        return df_id

    if group_data is None or group_data["groupby_column"] is None:
        df_dict = {"__all_data__": df}
    else:
        df_dict = {
            group: group_df
            for group, group_df in df.groupby(group_data["groupby_column"])
        }

    jobs = []
    for group, group_df in df_dict.items():
        jobs.append(
            delayed(_get_or_create_dataset_from_df_with_lock)(
                use_case_id=use_case_id,
                df=group_df,
                name=name,
                group=group,
            )
        )

    results = Parallel(n_jobs=10, backend="threading")(jobs)

    return_dict = {}
    for group, result in zip(df_dict.keys(), results):
        return_dict[group] = str(result)

    return return_dict, df_dict


def run_autopilot(
    token: str,
    endpoint: str,
    target_name: str,
    dataset_dict: Dict[str, str],
    use_case_id: str,
    experiment_config: Dict,
) -> Dict[str, str]:
    if use_case_id == "not_supported":
        use_case_id = None

    def _get_or_create_autopilot_run(
        name: str,
        use_case: str,
        dataset_id: str,
        advanced_options_config: Dict,
        analyze_and_model_config: Dict,
        create_from_dataset_config: Dict,
        datetime_partitioning_config: Dict,
        feature_settings_config: List[Dict],
    ) -> Optional[str]:
        try:
            project_id = get_or_create_autopilot_run(
                token=token,
                endpoint=endpoint,
                name=name,
                use_case=use_case_id,
                dataset_id=dataset_id,
                advanced_options_config=advanced_options_config,
                analyze_and_model_config=analyze_and_model_config,
                create_from_dataset_config=create_from_dataset_config,
                datetime_partitioning_config=datetime_partitioning_config,
                feature_settings_config=feature_settings_config,
            )
            return project_id
        except dr.errors.ClientError as e:
            log.error(f"Error creating project: {e}")
            return None

    return_dict = {}

    jobs = {}
    for group, dataset_id in dataset_dict.items():
        jobs[group] = []

        project_name = experiment_config["experiment_name"]
        if group != "__all_data__":
            project_name = f"{project_name} ({group})"
        log.info(f"Experiment Config: {experiment_config}")
        experiment_config["analyze_and_model"]["target"] = target_name

        jobs[group].append(
            delayed(_get_or_create_autopilot_run)(
                name=project_name,
                use_case=use_case_id,
                dataset_id=dataset_id,
                advanced_options_config=experiment_config.get("advanced_options", {}),
                analyze_and_model_config=experiment_config.get("analyze_and_model", {}),
                create_from_dataset_config=experiment_config.get(
                    "create_from_dataset", {}
                ),
                datetime_partitioning_config=experiment_config.get(
                    "datetime_partitioning", {}
                ),
                feature_settings_config=experiment_config.get("feature_settings", {}),
            )
        )

    results = Parallel(n_jobs=100, backend="threading")(
        job for group in jobs.keys() for job in jobs[group]
    )

    for group, result in zip(jobs.keys(), results):
        if result is not None:
            return_dict[group] = str(result)
    return return_dict


def unlock_holdouts(
    project_dict: Dict[str, str],
):
    for _, project_id in project_dict.items():
        project = dr.Project.get(project_id)
        project.unlock_holdout()
    return True


def merge_predictions(
    training_predictions: pd.DataFrame, training_data: pd.DataFrame
) -> pd.DataFrame:
    training_predictions = training_predictions.set_index("row_id")
    return training_data.join(training_predictions, how="inner")


def calculate_backtests(
    project_dict: Dict[str, str],
    holdouts_unlocked: bool,
    max_models_per_project: int = 5,
) -> bool:
    assert holdouts_unlocked, "Holdouts have not been unlocked"

    def _calculate_backtest(model: dr.DatetimeModel):
        # if rate_limiter.acquire():
        try:
            job = model.score_backtests()
            job.wait_for_completion()
        except dr.errors.ClientError as e:
            if e.json["message"] in [
                "All available backtests have already been scored.",
                "This job duplicates a job or jobs that are in the queue or have completed.",
            ]:
                # log.info(f"Backtests already calculated for model {model.id}")
                pass
            else:
                raise e
            # finally:
            #     rate_limiter.release()

    all_models = []
    for _, project_id in project_dict.items():
        project = dr.Project.get(project_id)
        models = get_models(project)[:max_models_per_project]
        all_models.extend(models)

    Parallel(n_jobs=100, backend="threading")(
        delayed(_calculate_backtest)(model) for model in all_models
    )
    # wait for all jobs on the project to complete
    jobs = []
    for project_id in project_dict.values():
        project = dr.Project.get(project_id)
        jobs.extend(project.get_all_jobs())

    log.info("Waiting for backtesting jobs to complete")
    wait_for_jobs(jobs, rate_limiter)

    return True


def get_backtest_predictions(
    project_dict: Dict[str, str],
    df_dict: Dict[str, pd.DataFrame],
    backtests_completed: bool,
    data_subset: Optional[
        Union[dr.enums.DATA_SUBSET, str]
    ] = dr.enums.DATA_SUBSET.ALL_BACKTESTS,
    max_models_per_project: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Get backtest predictions for each model in given projects using parallel processing.

    Args:
        project_dict: Dictionary mapping group names to project IDs.
        df_dict: Dictionary mapping group names to their respective DataFrames.
        backtests_completed: Flag to indicate whether backtests have been completed.
        data_subset: The subset of data to use for predictions.
        max_models_per_project: Maximum number of models to fetch predictions for from each project.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys formatted as "{group}/{model.project_id}/{model.id}/{partition_id}"
        and values as DataFrames with predictions.
    """

    if not backtests_completed:
        raise ValueError("Backtests have not been completed")

    def _get_backtest_predictions(
        model: dr.Model, group: str
    ) -> Dict[str, pd.DataFrame]:
        training_predictions = get_training_predictions(model, data_subset)
        training_data = df_dict[group].copy()
        merged_predictions = merge_predictions(training_predictions, training_data)
        return {
            f"{group}/{model.project_id}/{model.id}/{partition_id}": df
            for partition_id, df in merged_predictions.groupby("partition_id")
        }

    tasks = (
        delayed(_get_backtest_predictions)(model, group)
        for group, project_id in project_dict.items()
        for model in get_models(dr.Project.get(project_id))[:max_models_per_project]
    )

    results = Parallel(n_jobs=100, backend="threading")(tasks)

    aggregated_results = {}
    for result in results:
        aggregated_results.update(result)  # type: ignore

    return aggregated_results


def get_external_predictions(
    project_dict: Dict[str, str],
    external_holdout: pd.DataFrame,
    partition_column: Optional[str] = None,
    max_models_per_project: int = 5,
    group_data: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Retrieves external predictions for models across multiple projects, using external holdout data.

    Args:
        project_dict: Dictionary mapping group names to project IDs.
        external_holdout: DataFrame containing external data for predictions.
        partition_column: Optional column name to partition the data, defaults to None.
        max_models_per_project: Maximum number of models to consider per project.
        group_data: Optional dictionary specifying how to group data, default is None.

    Returns:
        Dictionary of formatted strings (group/model ID/partition ID) to their respective DataFrame of predictions.
    """

    def _get_external_predictions(
        model: dr.Model, prediction_df: pd.DataFrame, group: str
    ) -> Dict[str, pd.DataFrame]:
        external_prediction_df = get_external_holdout_predictions(model, prediction_df)
        merged_predictions = merge_predictions(external_prediction_df, prediction_df)
        # Default partition ID to "external_holdout" if the column is not found
        merged_predictions["partition_id"] = merged_predictions.get(
            partition_column, "external_holdout"
        )
        return {
            f"{group}/{model.project_id}/{model.id}/{partition_id}": df
            for partition_id, df in merged_predictions.groupby("partition_id")
        }

    # Group data based on specified group data or use the whole data as a single group
    log.info(f"Group data: {group_data}")
    if (
        group_data is not None
        and "groupby_column" in group_data
        and group_data["groupby_column"] is not None
    ):
        df_dict = {
            group: group_df
            for group, group_df in external_holdout.groupby(
                group_data["groupby_column"]
            )
        }
    else:
        df_dict = {"__all_data__": external_holdout}

    # Fetch models and prepare tasks for parallel processing
    tasks = (
        delayed(_get_external_predictions)(model, df_dict[group], group)
        for group, project_id in project_dict.items()
        for model in get_models(dr.Project.get(project_id))[:max_models_per_project]
    )

    results = Parallel(n_jobs=100, backend="threading")(tasks)

    # Flatten results and combine into a single dictionary
    aggregated_results = {}
    for result in results:
        aggregated_results.update(result)  # type: ignore

    return aggregated_results


def get_datarobot_metrics(
    project_dict: Dict[str, str],
):
    for group, project_id in project_dict.items():
        project = dr.Project.get(project_id)

        for model in get_models(project):
            model.metrics
