"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.19.3
"""

from typing import Any
import pandas as pd
from pydantic import BaseModel, Field, field_validator

import logging

log = logging.getLogger(__name__)


class Operation(BaseModel):
    operation: str = Field(...)

    @field_validator("operation")
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


def binarize_data(
    input_data: pd.DataFrame,
    target: str,
    threshold=0.0,
    drop_regression_target=True,
    operation: Operation = Operation(operation=">"),
    new_target_name="target_cat",
) -> tuple[pd.DataFrame, str]:
    """helper function: binarize a target variable for classification"""
    categorical_data = input_data.copy()

    op = operation.operation

    if op == ">":
        categorical_data[new_target_name] = categorical_data[target] > threshold
    elif op == "<":
        categorical_data[new_target_name] = categorical_data[target] < threshold
    elif op == ">=":
        categorical_data[new_target_name] = categorical_data[target] >= threshold
    elif op == "<=":
        categorical_data[new_target_name] = categorical_data[target] <= threshold
    elif op == "==":
        categorical_data[new_target_name] = categorical_data[target] == threshold
    elif op == "!=":
        categorical_data[new_target_name] = categorical_data[target] != threshold
    else:
        categorical_data[new_target_name] = categorical_data[target] > threshold
        log.warning("Unrecognised operation. Defaulting to >")

    if drop_regression_target:
        categorical_data.drop(columns=[target], inplace=True)

    return categorical_data, new_target_name
