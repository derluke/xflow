"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.19.3
"""

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional, Tuple, Union

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd
from datarobotx.idp.autopilot import get_or_create_autopilot_run
from datarobotx.idp.common.hashing import get_hash
from datarobotx.idp.datasets import get_or_create_dataset_from_df
from filelock import FileLock
from joblib import Parallel, delayed
from pydantic import BaseModel, Field, field_validator
from utils.dr_helpers import (
    get_external_holdout_predictions,
    get_models,
    get_training_predictions,
    wait_for_jobs,
)

max_calls = 20
semaphore = threading.Semaphore(max_calls)

log = logging.getLogger(__name__)

dr_logger = logging.getLogger("datarobot.models.project")
for handler in dr_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        # log.info(f"Removing handler {handler}")
        dr_logger.removeHandler(handler)


class Operator(BaseModel):
    operator: str = Field(...)

    @field_validator("operator")
    def normalize_operation(cls, v):
        operation_mapping = {
            "<": ["lt", "less than", "less"],
            ">": ["gt", "greater than", "greater"],
            "<=": ["lte", "less than or equal to", "less equal"],
            ">=": ["gte", "greater than or equal to", "greater equal"],
            "==": ["eq", "equal to", "equals", "equal"],
            "!=": ["ne", "not equal to", "not equals", "not equal"],
        }

        for op, aliases in operation_mapping.items():
            if v.lower() in aliases + [op]:
                return op

        raise ValueError(
            f"""Invalid operation: {v}
            allowed Values:
            {operation_mapping}
        """
        )

    def apply_operation(self, threshold: float) -> Callable[[float], bool]:
        return {
            ">": lambda x: x > threshold,
            "<": lambda x: x < threshold,
            ">=": lambda x: x >= threshold,
            "<=": lambda x: x <= threshold,
            "==": lambda x: x == threshold,
            "!=": lambda x: x != threshold,
        }[self.operator]


def unpack_row_to_args(
    control_series: dict, arg_look: dict, arg_values_dict=None, verbose=False
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
    input_data: pd.DataFrame, target: str, binarize_data_config: dict
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
    group_data: Optional[dict] = None,
) -> Tuple[Dict[str, str], Dict[str, pd.DataFrame]]:
    # log.warning(f"Grouping data by {group_data['groupby_column']} for experiment {name}")

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
    experiment_config: dict,
) -> Dict[str, str]:
    use_case = dr.UseCase.get(use_case_id)

    def _get_or_create_autopilot_run(
        name: str,
        use_case: str,
        dataset_id: str,
        advanced_options_config: dict,
        analyze_and_model_config: dict,
        create_from_dataset_config: dict,
        datetime_partitioning_config: dict,
        feature_settings_config: dict,
    ) -> Optional[str]:
        try:
            project_id = get_or_create_autopilot_run(
                token=token,
                endpoint=endpoint,
                name=name,
                use_case=use_case,
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
        dataset = dr.Dataset.get(dataset_id)

        project_name = experiment_config.get(
            "experiment_name", f"{use_case.name}:{target_name} [{dataset.name}]"
        )
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

    results = Parallel(n_jobs=10, backend="threading")(
        job for group in jobs.keys() for job in jobs[group]
    )

    for group, result in zip(jobs.keys(), results):
        if result is not None:
            return_dict[group] = str(result)
    return return_dict


def merge_predictions(
    training_predictions: pd.DataFrame, training_data: pd.DataFrame
) -> pd.DataFrame:
    training_predictions = training_predictions.set_index("row_id")
    return training_data.join(training_predictions, how="inner")


def calculate_backtests(
    project_dict: Dict[str, str], max_models_per_project: int = 5
) -> bool:

    def _calculate_backtest(model: dr.DatetimeModel):
        with semaphore:
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

    all_models = []
    for _, project_id in project_dict.items():
        project = dr.Project.get(project_id)
        models = get_models(project)[:max_models_per_project]
        all_models.extend(models)

    Parallel(n_jobs=10, backend="threading")(
        delayed(_calculate_backtest)(model) for model in all_models
    )
    # wait for all jobs on the project to complete
    jobs = []
    for project_id in project_dict.values():
        project = dr.Project.get(project_id)
        jobs.extend(project.get_all_jobs())

    log.info("Waiting for backtesting jobs to complete")
    wait_for_jobs(jobs)

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
    Get backtest predictions for a project.

    Args:
        project_id: Project ID.
        df: Dataframe.
        backtests_completed: Whether backtests have been completed.
        data_subset: Data subset.
        max_models_per_project: Maximum number of models per project.

    Returns:
        Dict[str, List[Dict[str, pd.DataFrame]]]: Backtest predictions.
    """

    def _get_backtest_predictions(
        model: dr.Model, data_subset: dr.enums.DATA_SUBSET, group: str
    ) -> Dict[str, pd.DataFrame]:
        with semaphore:
            training_predictions = get_training_predictions(model, data_subset)
            training_data = df_dict[group].copy()
            merged_predictions = merge_predictions(training_predictions, training_data)
            backtest_dict = {
                f"{group}/{model.id}/{partition_id}": df
                for partition_id, df in merged_predictions.groupby("partition_id")
            }
            return backtest_dict

    if not backtests_completed:
        raise ValueError("Backtests have not been completed")

    all_models = {}
    for group, project_id in project_dict.items():
        project = dr.Project.get(project_id)
        models = get_models(project)[:max_models_per_project]
        all_models[group] = models

    results = Parallel(n_jobs=10, backend="threading")(
        delayed(_get_backtest_predictions)(model, data_subset, group)
        for group in all_models.keys()
        for model in all_models[group]
    )

    return {f"{project.id}/{k}": df for row in results for k, df in row.items()}  # type: ignore


def get_external_predictions(
    project_dict: Dict[str, str],
    external_holdout: pd.DataFrame,
    partition_column: Optional[str] = None,
    max_models_per_project: int = 5,
    group_data: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Get external predictions for a project.

    Args:
        project_id: Project ID.
        external_holdout: External holdout dataframe.
        max_models_per_project: Maximum number of models per project.

    Returns:
        Dict[str, List[Dict[str, pd.DataFrame]]]: External predictions.
    """

    def _get_external_predictions(
        model: dr.Model,
        prediction_df: pd.DataFrame,
        partition_column: Optional[str],
        group: str,
    ) -> Dict[str, pd.DataFrame]:
        with semaphore:
            external_prediction_df = get_external_holdout_predictions(
                model, prediction_df
            )
            merged_predictions = merge_predictions(
                external_prediction_df, prediction_df
            )
            if partition_column not in merged_predictions.columns:
                merged_predictions["partition_id"] = "external_holdout"
            external_predictions_dict = {
                f"{group}/{model.id}/{partition_id}": df
                for partition_id, df in merged_predictions.groupby("partition_id")
            }
            return external_predictions_dict  # type: ignore

    if group_data is None or group_data["groupby_column"] is None:
        df_dict = {"__all_data__": external_holdout}
    else:
        df_dict = {
            group: group_df
            for group, group_df in external_holdout.groupby(
                group_data["groupby_column"]
            )
        }

    all_models = {}
    for group, project_id in project_dict.items():
        project = dr.Project.get(project_id)
        models = get_models(project)[:max_models_per_project]
        all_models[group] = models

    results = Parallel(n_jobs=10, backend="threading")(
        delayed(_get_external_predictions)(
            model, df_dict[group], partition_column, group
        )
        for group in all_models.keys()
        for model in all_models[group]
    )

    return {f"{project.id}/{k}": df for row in results for k, df in row.items()}  # type: ignore
