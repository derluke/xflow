from typing import Dict

import pandas as pd
from sklearn.metrics import mean_squared_error

from .metrics import Metric


class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__(
            data_type="continuous",
            name="mean_squared_error",
            higher_is_better={"mean_squared_error": False},
        )

    def compute(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        **kwargs,
    ) -> Dict[str, float]:
        return {self.name: mean_squared_error(actuals, predictions)}  # type: ignore
