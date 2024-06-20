from typing import Any, Dict, Optional

import pandas as pd
from sklearn.metrics import mean_squared_error

from .metrics import Metric


class MeanSquaredError(Metric):
    def __init__(self) -> None:
        super().__init__(
            data_type="continuous",
            name="mean_squared_error",
            higher_is_better={"mean_squared_error": False},
        )

    def compute(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        extra_data: Optional[pd.DataFrame],
        experiment_config: Optional[Dict[str, Any]],
        metric_config: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> dict[str, Optional[float]]:
        return {self.name: mean_squared_error(actuals, predictions)}
