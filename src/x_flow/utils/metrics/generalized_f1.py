from typing import Any, Dict, Optional

import pandas as pd
from sklearn.metrics import f1_score

from .metrics import Metric
import logging

log = logging.getLogger(__name__)


class GeneralizedF1(Metric):
    def __init__(self) -> None:
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
        experiment_config: dict[str, Any],
        metric_config: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> dict[str, Optional[float]]:
        # log.info(f"Computing generalized F1 score with metric_config: {metric_config}")

        if metric_config is None:
            metric_config = {}

        # check if actuals have only 2 unique values
        if actuals.nunique() == 2:
            binary_model = True
        else:
            binary_model = False

        if not binary_model:
            log.info(metric_config)
            predictions = self.preprocess(
                predictions,
                experiment_config=experiment_config,
                metric_config=metric_config,
            )
            actuals = self.preprocess(
                actuals,
                experiment_config=experiment_config,
                metric_config=metric_config,
            )
        else:
            threshold = metric_config.get("threshold", 0.5)
            predictions = predictions > threshold

        predictions = predictions.astype(bool)

        return {self.name: float(f1_score(actuals, predictions))}
