"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.19.3
"""

import logging
import tempfile
from typing import Any, Optional

# pyright: reportPrivateImportUsage=false
import datarobot as dr
import pandas as pd
from datarobotx.idp.autopilot import get_or_create_autopilot_run
from datarobotx.idp.common.hashing import get_hash
from datarobotx.idp.datasets import get_or_create_dataset_from_df
from filelock import FileLock
from pydantic import BaseModel, Field, field_validator

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

    with FileLock(f"{df_token}.lock"):
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
