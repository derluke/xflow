from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

from .metrics import Metric


class GeneralizedF1(Metric):
    def __init__(self):
        super().__init__(data_type="binary")

    def compute(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        extra_data: Optional[pd.DataFrame],
        experiment_config: Optional[dict],
        metric_config: Optional[dict],
    ) -> float:
        predictions = self.preprocess(
            predictions,
            experiment_config=experiment_config,
            metric_config=metric_config,
        )
        actuals = actuals.astype(bool)
        predictions = predictions.astype(bool)
        return float(f1_score(actuals, predictions))


class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__(data_type="continuous")

    def compute(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        extra_data: Optional[pd.DataFrame],
        experiment_config: Optional[dict],
        metric_config: Optional[dict],
    ) -> float:
        return mean_squared_error(actuals, predictions)
