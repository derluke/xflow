import polars as pl

from ..data import Data
from ..operator import Operator
from .data_preprocessor import DataPreprocessor


class BinarizeData(DataPreprocessor):
    def __init__(
        self,
        threshold: float,
        operator: str,
        binarize_drop_regression_target=True,
        binarize_new_target_name="target_cat",
    ):
        self._threshold = threshold
        self._operator = operator
        self._binarize_drop_regression_target = binarize_drop_regression_target
        self._binarize_new_target_name = binarize_new_target_name

    def _fit(self, df: Data):
        return self

    def _transform(self, df: Data) -> Data:
        """helper function: binarize a target variable for classification"""
        categorical_data = df.rendered_df
        target_series = categorical_data[df.target_column]

        op_fun = Operator(operator=self._operator).apply_operation(self._threshold)

        categorical_data = categorical_data.with_columns(
            target_series.map_elements(op_fun, return_dtype=pl.Boolean).alias(
                self._binarize_new_target_name
            )
        )
        if self._binarize_drop_regression_target:
            categorical_data.drop(df.target_column)

        df.df = categorical_data
        df.target_column = self._binarize_new_target_name

        return df
