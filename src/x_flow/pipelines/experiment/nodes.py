"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.19.3
"""

import logging
import os
from typing import Any, Dict, Optional, Union

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
    op = Operator(operator=binarize_operator).operator

    if op == ">":
        categorical_data[binarize_new_target_name] = (
            categorical_data[target] > binarize_threshold
        )
    elif op == "<":
        categorical_data[binarize_new_target_name] = (
            categorical_data[target] < binarize_threshold
        )
    elif op == ">=":
        categorical_data[binarize_new_target_name] = (
            categorical_data[target] >= binarize_threshold
        )
    elif op == "<=":
        categorical_data[binarize_new_target_name] = (
            categorical_data[target] <= binarize_threshold
        )
    elif op == "==":
        categorical_data[binarize_new_target_name] = (
            categorical_data[target] == binarize_threshold
        )
    elif op == "!=":
        categorical_data[binarize_new_target_name] = (
            categorical_data[target] != binarize_threshold
        )
    else:
        categorical_data[binarize_new_target_name] = (
            categorical_data[target] > binarize_threshold
        )
        log.warning("Unrecognised operation. Defaulting to >")

    if binarize_drop_regression_target:
        categorical_data.drop(columns=[target], inplace=True)

    return categorical_data, binarize_new_target_name


def binarize_data_node(
    input_data: pd.DataFrame, target: str, binarize_data_config: dict
) -> tuple[pd.DataFrame, str]:
    # log.info(f"input_data: {input_data}")
    log.info(f"target: {target}")
    log.info(f"binarize_data_config: {binarize_data_config}")
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
) -> str:
    df_token = get_hash(df, use_case_id, name)

    with FileLock(
        os.path.join(".locks", f"get_or_create_dataset_from_df_{df_token}.lock")
    ):
        return get_or_create_dataset_from_df(
            token=token,
            endpoint=endpoint,
            use_cases=use_case_id,
            data_frame=df,
            name=name,
        )


def run_autopilot(
    token: str,
    endpoint: str,
    target_name: str,
    dataset_id: str,
    use_case_id: str,
    experiment_config: dict,
) -> str:
    use_case = dr.UseCase.get(use_case_id)
    dataset = dr.Dataset.get(dataset_id)

    project_name = experiment_config.get(
        "experiment_name", f"{use_case.name}:{target_name} [{dataset.name}]"
    )
    log.info(f"Experiment Config: {experiment_config}")
    experiment_config["analyze_and_model"]["target"] = target_name

    return str(
        get_or_create_autopilot_run(
            token=token,
            endpoint=endpoint,
            name=project_name,
            use_case=use_case_id,
            dataset_id=dataset_id,
            advanced_options_config=experiment_config.get("advanced_options", {}),
            analyze_and_model_config=experiment_config.get("analyze_and_model", {}),
            create_from_dataset_config=experiment_config.get("create_from_dataset", {}),
            datetime_partitioning_config=experiment_config.get(
                "datetime_partitioning", {}
            ),
            feature_settings_config=experiment_config.get("feature_settings", {}),
        )
    )


def merge_predictions(
    training_predictions: pd.DataFrame, training_data: pd.DataFrame
) -> pd.DataFrame:
    training_predictions = training_predictions.set_index("row_id")
    return training_data.join(training_predictions, how="inner")


def calculate_backtests(project_id: str, max_models_per_project: int = 5) -> bool:
    project = dr.Project.get(project_id)
    models = get_models(project)[:max_models_per_project]

    def _calculate_backtest(model: dr.DatetimeModel):
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

    Parallel(n_jobs=10, backend="threading")(
        delayed(_calculate_backtest)(model) for model in models
    )
    # wait for all jobs on the project to complete
    jobs = project.get_all_jobs()
    log.info("Waiting for backtesting jobs to complete")
    wait_for_jobs(jobs)

    return True


def get_backtest_predictions(
    project_id: str,
    df: pd.DataFrame,
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
    if not backtests_completed:
        raise ValueError("Backtests have not been completed")

    project = dr.Project.get(project_id)
    models = get_models(project)[:max_models_per_project]

    def _get_backtest_predictions(
        model: dr.Model, data_subset: dr.enums.DATA_SUBSET
    ) -> Dict[str, pd.DataFrame]:
        training_predictions = get_training_predictions(model, data_subset)
        training_data = df.copy()
        merged_predictions = merge_predictions(training_predictions, training_data)
        backtest_dict = {
            f"{model.id}/{partition_id}": df
            for partition_id, df in merged_predictions.groupby("partition_id")
        }
        return backtest_dict

    results = Parallel(n_jobs=10, backend="threading")(
        delayed(_get_backtest_predictions)(model, data_subset) for model in models
    )

    return {f"{project.id}/{k}": df for row in results for k, df in row.items()}  # type: ignore


def get_external_predictions(
    project_id: str,
    external_holdout: pd.DataFrame,
    partition_column: Optional[str] = None,
    max_models_per_project: int = 5,
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
    project = dr.Project.get(project_id)
    models = get_models(project)[:max_models_per_project]

    def _get_external_predictions(
        model: dr.Model, prediction_df: pd.DataFrame, partition_column: Optional[str]
    ) -> Dict[str, pd.DataFrame]:

        external_prediction_df = get_external_holdout_predictions(model, prediction_df)
        merged_predictions = merge_predictions(external_prediction_df, prediction_df)
        if partition_column not in merged_predictions.columns:
            merged_predictions["partition_id"] = "external_holdout"
        external_predictions_dict = {
            f"{model.id}/{partition_id}": df
            for partition_id, df in merged_predictions.groupby("partition_id")
        }
        return external_predictions_dict  # type: ignore

    results = Parallel(n_jobs=10, backend="threading")(
        delayed(_get_external_predictions)(model, external_holdout, partition_column)
        for model in models
    )

    return {f"{project.id}/{k}": df for row in results for k, df in row.items()}  # type: ignore
