from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import f1_score

from .metrics import Metric


class GeneralizedF1(Metric):
    def __init__(self):
        super().__init__(
            data_type="binary",
            name="generalized_f1",
            higher_is_better={"generalized_f1": True},
        )

    def compute(  # noqa: PLR0913
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        extra_data: Optional[pd.DataFrame],
        experiment_config: Optional[dict],
        metric_config: Optional[dict],
        **kwargs,
    ) -> Dict[str, float]:
        # log.info(f"Computing generalized F1 score with metric_config: {metric_config}")

        if metric_config is None:
            metric_config = {}

        # check if actuals have only 2 unique values
        if actuals.nunique() == 2:
            binary_model = True
        else:
            binary_model = False

        if not binary_model:
            predictions = self.preprocess(
                predictions,
                experiment_config=experiment_config,  # type: ignore
                metric_config=metric_config,  # type: ignore
            )
            actuals = self.preprocess(
                actuals,
                experiment_config=experiment_config,  # type: ignore
                metric_config=metric_config,  # type: ignore
            )
        else:
            threshold = metric_config.get("threshold", 0.5)
            predictions = predictions > threshold
        predictions = predictions.astype(bool)
        # log.info(actuals.head())
        # log.info(predictions.head())
        return {self.name: float(f1_score(actuals, predictions))}
