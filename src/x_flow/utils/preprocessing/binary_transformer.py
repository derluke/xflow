from ..data import Data, TrainingData
from ..operator import Operator
from .data_preprocessor import DataPreprocessor


class BinarizeData(DataPreprocessor):
    def __init__(
        self,
        binarize_threshold: float,
        binarize_operator: str,
        binarize_drop_regression_target=True,
        binarize_new_target_name="target_cat",
    ):
        self._threshold = binarize_threshold
        self._operator = binarize_operator
        self._binarize_drop_regression_target = binarize_drop_regression_target
        self._binarize_new_target_name = binarize_new_target_name

    def _fit(self, df: Data):
        return self

    def _transform(self, df: TrainingData) -> Data:
        """Helper function: binarize a target variable for classification"""
        categorical_data = df.rendered_df
        target_series = categorical_data[df.target_column]

        op_fun = Operator(operator=self._operator).apply_operation(self._threshold)

        # Apply the operation and create a new boolean column
        categorical_data[self._binarize_new_target_name] = target_series.apply(op_fun).astype(bool)

        # Optionally drop the original target column
        if self._binarize_drop_regression_target:
            categorical_data = categorical_data.drop(columns=[df.target_column])

        df.df = categorical_data
        df.target_column = self._binarize_new_target_name

        return df
