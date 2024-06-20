from dataclasses import asdict
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from filelock import FileLock
from joblib import Parallel, delayed
import pandas as pd
import requests

from x_flow.utils.data import (
    Data,
    TrainingData,
    ValidationData,
    ValidationPredictionData,
)
from x_flow.utils.dr_helpers import (
    RateLimiterSemaphore,
    get_external_holdout_predictions,
    get_models,
    get_training_predictions,
    wait_for_jobs,
)
from x_flow.utils.preprocessing.binary_transformer import BinarizeData
from x_flow.utils.preprocessing.data_preprocessor import DataPreprocessor, Identity
from x_flow.utils.preprocessing.fire import FIRE

# pyright: reportPrivateImportUsage=false
import datarobot as dr
from datarobot.rest import RESTClientObject

from datarobotx.idp.autopilot import get_or_create_autopilot_run
from datarobotx.idp.common.hashing import get_hash
from datarobotx.idp.datasets import get_or_create_dataset_from_df
from datarobotx.idp.use_cases import get_or_create_use_case

try:
    from datarobot import UseCase  # type: ignore
except:

    class UseCase:  # type: ignore[no-redef]
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
rate_limiter = RateLimiterSemaphore(25)

client_request_fun = RESTClientObject.request


def request_with_rate_limiter(*args: Any, **kwargs: Any) -> requests.Response:
    rate_limiter.acquire()
    try:
        response = client_request_fun(*args, **kwargs)
    except dr.errors.ClientError as e:
        # log.error(f"Error in request: {e}")
        if (
            e.json["message"]
            == "You have exceeded your limit on total modeling API requests.  Try again in 1 seconds."
        ):
            log.warning("Rate limit exceeded, waiting for 5 seconds")
            time.sleep(5)
            response = client_request_fun(*args, **kwargs)
        else:
            raise e
    rate_limiter.release()
    return response


RESTClientObject.request = request_with_rate_limiter  # type: ignore[method-assign]

log = logging.getLogger(__name__)

dr_logger = logging.getLogger("datarobot.models.project")
for handler in dr_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        # log.info(f"Removing handler {handler}")
        dr_logger.removeHandler(handler)


def unpack_row_to_args(
    control_series: dict[str, Any],
    arg_look: Dict[str, Any],
    arg_values_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    # Your code here
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


# def binarize_data_if_specified(  # noqa: PLR0913
#     input_data: pd.DataFrame,
#     target: str,
#     binarize_threshold=0.0,
#     binarize_drop_regression_target=True,
#     binarize_operator: Optional[str] = None,
#     binarize_new_target_name="target_cat",
# ) -> tuple[pd.DataFrame, str]:
#     """helper function: binarize a target variable for classification"""
#     if binarize_operator is None:
#         return input_data, target

#     transformer = BinarizeData(
#         threshold=binarize_threshold,
#         operator=binarize_operator,
#         binarize_drop_regression_target=binarize_drop_regression_target,
#         binarize_new_target_name=binarize_new_target_name,
#     )
#     return transformer.fit_transform(input_data)
#     categorical_data = input_data.copy()

#     op_fun = Operator(operator=binarize_operator).apply_operation(binarize_threshold)

#     categorical_data[binarize_new_target_name] = categorical_data[target].apply(op_fun)
#     if binarize_drop_regression_target:
#         categorical_data.drop(columns=[target], inplace=True)

#     return categorical_data, binarize_new_target_name


def preprocessing_fit_transform(data: Data, *transformations: DataPreprocessor) -> Data:
    for transformation in transformations:
        data = transformation.fit_transform(data)
    return data


def preprocessing_transform(data: Data, *transformations: DataPreprocessor) -> Data:
    for transformation in transformations:
        data = transformation.transform(data)
    return data


def register_binarize_preprocessor(binarize_data_config: dict[str, Any]) -> DataPreprocessor:
    if binarize_data_config is None:
        transformer = Identity()
    else:
        transformer = BinarizeData(**binarize_data_config)
    return transformer


def register_fire_preprocessor(fire_config: dict[str, Any]) -> DataPreprocessor:
    if fire_config is None:
        transformer = Identity()
    else:
        transformer = FIRE(**fire_config)
    return transformer


def get_or_create_use_case_with_lock(
    token: str,
    endpoint: str,
    name: str,
) -> str:
    with FileLock(os.path.join(".locks", f"get_or_create_use_case_{name}.lock")):
        use_case_id = get_or_create_use_case(token=token, endpoint=endpoint, name=name)
    return use_case_id


def get_or_create_dataset_from_df_with_lock(
    token: str,
    endpoint: str,
    use_case_id: str,
    df: Data,
    name: str,
) -> Dict[str, str]:
    # log.warning(f"Grouping data by {group_data['groupby_column']} for experiment {name}")

    if use_case_id == "not_supported":
        use_case_id = None  # type: ignore

    def _get_or_create_dataset_from_df_with_lock(
        use_case_id: str,
        df: pd.DataFrame,
        name: str,
        group: Optional[str] = None,
    ) -> str:
        name = f"{name}_{group}" if (group != "__all_data__") else name
        df_token = get_hash(df, use_case_id, name)

        with FileLock(os.path.join(".locks", f"get_or_create_dataset_from_df_{df_token}.lock")):
            df_id = get_or_create_dataset_from_df(
                token=token,
                endpoint=endpoint,
                use_cases=use_case_id,
                data_frame=df,
                name=name,
            )
        return df_id

    group_data = df.get_partitions()
    jobs = []
    for group, group_df in group_data.items():
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
    for group, result in zip(group_data.keys(), results):
        return_dict[group] = str(result)

    return return_dict


def run_autopilot(  # noqa: PLR0913
    token: str,
    endpoint: str,
    df: TrainingData,
    dataset_dict: dict[str, str],
    use_case_id: str,
    experiment_config: dict[str, Any],
) -> Dict[str, str]:
    if use_case_id == "not_supported":
        use_case_id = None  # type: ignore

    def _get_or_create_autopilot_run(  # noqa: PLR0913
        name: str,
        use_case: str,
        dataset_id: str,
        advanced_options_config: dict[str, Any],
        analyze_and_model_config: dict[str, Any],
        create_from_dataset_config: dict[str, Any],
        datetime_partitioning_config: Optional[dict[str, Any]],
        feature_settings_config: list[dict[str, Any]],
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

    jobs: dict[str, List[Any]] = {}
    for group, dataset_id in dataset_dict.items():
        jobs[group] = []

        project_name = experiment_config["experiment_name"]
        if group != "__all_data__":
            project_name = f"{project_name} ({group})"
        # log.info(f"Experiment Config: {experiment_config}")
        experiment_config["analyze_and_model"]["target"] = df.target_column

        jobs[group].append(
            delayed(_get_or_create_autopilot_run)(
                name=project_name,
                use_case=use_case_id,
                dataset_id=dataset_id,
                advanced_options_config=experiment_config.get("advanced_options", {}),
                analyze_and_model_config=experiment_config.get("analyze_and_model", {}),
                create_from_dataset_config=experiment_config.get("create_from_dataset", {}),
                datetime_partitioning_config=experiment_config.get("datetime_partitioning", {}),
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
    project_dict: dict[str, str],
) -> bool:
    for _, project_id in project_dict.items():
        project = dr.Project.get(project_id)  # type: ignore[attr-defined]
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

    def _calculate_backtest(model: dr.DatetimeModel) -> None:
        try:
            job = model.score_backtests()
            job.wait_for_completion()
        except dr.errors.ClientError as e:
            if e.json["message"] in [
                "All available backtests have already been scored.",
                "This job duplicates a job or jobs that are in the queue or have completed.",
            ]:
                pass
            else:
                raise e

    all_models: List[dr.DatetimeModel] = []
    for _, project_id in project_dict.items():
        project = dr.Project.get(project_id)  # type: ignore[attr-defined]
        models = get_models(project)[:max_models_per_project]
        all_models.extend(models)  # type: ignore[arg-type]

    Parallel(n_jobs=100, backend="threading")(
        delayed(_calculate_backtest)(model) for model in all_models
    )
    # wait for all jobs on the project to complete
    jobs = []
    for project_id in project_dict.values():
        project = dr.Project.get(project_id)  # type: ignore[attr-defined]
        jobs.extend(project.get_all_jobs())

    log.info("Waiting for backtesting jobs to complete")
    wait_for_jobs(jobs, rate_limiter)

    return True


def get_backtest_predictions(
    project_dict: Dict[str, str],
    df: ValidationData,
    # backtests_completed: bool,
    data_subset: Optional[Union[dr.enums.DATA_SUBSET, str]] = dr.enums.DATA_SUBSET.ALL_BACKTESTS,
    max_models_per_project: int = 5,
) -> Dict[str, ValidationPredictionData]:
    """
    Get backtest predictions for each model in given projects using parallel processing.

    Args:
        project_dict: Dictionary mapping group names to project IDs.
        df_dict: Dictionary mapping group names to their respective DataFrames.
        backtests_completed: Flag to indicate whether backtests have been completed.
        data_subset: The subset of data to use for predictions.
        max_models_per_project: Maximum number of models to fetch predictions for from each project.

    Returns
    -------
        Dict[str, pd.DataFrame]: Dictionary with keys formatted as "{group}/{model.project_id}/{model.id}/{partition_id}"
        and values as DataFrames with predictions.
    """
    # if not backtests_completed:
    #     raise ValueError("Backtests have not been completed")

    def _get_backtest_predictions(
        model: dr.Model, group: str
    ) -> Dict[str, ValidationPredictionData]:
        training_predictions = get_training_predictions(model, data_subset)  # type: ignore
        training_data = df.get_partitions()[group].copy()
        merged_predictions = merge_predictions(training_predictions, training_data)  # type: ignore

        result_dict = {}
        for partition, group_df in merged_predictions.groupby("partition_id"):
            result = ValidationPredictionData(**asdict(df))
            result.df = group_df
            result_dict[f"{group}/{model.project_id}/{model.id}/{partition}"] = result

        return result_dict

    tasks = (
        delayed(_get_backtest_predictions)(model, group)
        for group, project_id in project_dict.items()
        for model in get_models(dr.Project.get(project_id))[:max_models_per_project]  # type: ignore[attr-defined]
    )

    results = Parallel(n_jobs=100, backend="threading")(tasks)

    aggregated_results = {}
    for result in results:
        aggregated_results.update(result)  # type: ignore

    return aggregated_results


def get_external_predictions(
    project_dict: Dict[str, str],
    external_holdout: ValidationData,
    max_models_per_project: int = 5,
) -> Dict[str, ValidationPredictionData]:
    """
    Retrieve external predictions for models across multiple projects, using external holdout data.

    Args:
        project_dict: Dictionary mapping group names to project IDs.
        external_holdout: DataFrame containing external data for predictions.
        partition_column: Optional column name to partition the data, defaults to None.
        max_models_per_project: Maximum number of models to consider per project.
        group_data: Optional dictionary specifying how to group data, default is None.

    Returns
    -------
        Dictionary of formatted strings (group/model ID/partition ID) to their respective DataFrame of predictions.
    """

    def _get_external_predictions(
        model: dr.Model, prediction_df: pd.DataFrame, group: str
    ) -> dict[str, ValidationPredictionData]:
        external_prediction_df = get_external_holdout_predictions(model, prediction_df)
        merged_predictions = merge_predictions(external_prediction_df, prediction_df)

        result = ValidationPredictionData(**asdict(external_holdout))
        result.df = merged_predictions
        return {f"{group}/{model.project_id}/{model.id}/external_holdout": result}

    df_dict = external_holdout.get_partitions()

    # Fetch models and prepare tasks for parallel processing
    tasks = (
        delayed(_get_external_predictions)(model, df_dict[group], group)
        for group, project_id in project_dict.items()
        for model in get_models(dr.Project.get(project_id))[:max_models_per_project]  # type: ignore[attr-defined]
    )

    results = Parallel(n_jobs=100, backend="threading")(tasks)

    # Flatten results and combine into a single dictionary
    aggregated_results = {}
    for result in results:
        aggregated_results.update(result)  # type: ignore

    return aggregated_results
