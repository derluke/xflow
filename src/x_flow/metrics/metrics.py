import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd
from utils.operator import Operator

log = logging.getLogger(__name__)

# TODO:
# lower is better attribute (P1)
# display name (nicer name) (P1)
# support multiple thresholds (P2)
# aggregation method: sum, mean, median, max, min, etc. gauge


class Metric(ABC):
    def __init__(self, data_type: str):
        self.data_type = data_type  # 'binary' or 'continuous'

    @abstractmethod
    def compute(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        extra_data: Optional[pd.DataFrame],
        experiment_config: Optional[Dict],
        metric_config: Optional[Dict],
    ) -> float:
        pass

    def preprocess(
        self, predictions: pd.Series, experiment_config: dict, metric_config: dict
    ) -> pd.Series:
        """
        Preprocesses the predictions based on the specified data type of the metric, particularly for binary classification.

        This method handles the binarization of predictions if the metric requires binary data. Binarization parameters
        can be specified in either the experiment configuration or the metric configuration. The method first checks for
        binarization settings in the experiment configuration. If these settings are not fully specified, it then looks
        for them in the metric configuration.

        Args:
            predictions (pd.Series): The predictions to be preprocessed, typically as a Pandas Series.
            experiment_config (dict): Configuration dictionary that may contain binarization settings under the key
                                    'binarize_data'. It should have sub-keys 'binarize_operator' and 'binarize_threshold'.
            metric_config (dict): Fallback configuration dictionary that should be checked if 'experiment_config' does not
                                fully specify binarization. It must contain a 'binarize_data_config' key with sub-keys
                                'operator' and 'threshold' for binarization.

        Returns:
            pd.Series: The preprocessed predictions, which may be binarized if specified by the configurations. If the
                    metric's data type is not 'binary', the predictions are returned unchanged.

        Raises:
            ValueError: If the necessary binarization parameters are missing in both configurations when required,
                        or if the 'experiment_config' or 'metric_config' itself is not provided when needed.
        """
        if self.data_type == "binary":
            if experiment_config is None:
                raise ValueError("experiment_config is required")

            binarize_data = experiment_config.get("binarize_data", {})
            binarize_operator = binarize_data.get("binarize_operator")
            binarize_threshold = binarize_data.get("binarize_threshold")

            if not binarize_operator or binarize_threshold is None:
                if metric_config is None or "binarize_data_config" not in metric_config:
                    raise ValueError(
                        "metric_config is required with 'binarize_data_config' key when not specified in experiment_config."
                    )

                binarize_data = metric_config["binarize_data_config"]
                binarize_operator = binarize_data.get("operator")
                binarize_threshold = binarize_data.get("threshold")

                if binarize_operator is None or binarize_threshold is None:
                    raise ValueError(
                        "Both 'operator' and 'threshold' must be specified in metric_config."
                    )

                if binarize_operator and binarize_threshold is not None:
                    op_fun = Operator(operator=binarize_operator).apply_operation(
                        binarize_threshold
                    )
                return op_fun(predictions)

        return predictions


class MetricFactory:
    metrics = {}

    @classmethod
    def register_metric(cls, name: str, metric: Metric):
        cls.metrics[name] = metric

    @classmethod
    def get_metric(cls, name: str) -> Metric:
        metric = cls.metrics.get(name)
        if not metric:
            raise ValueError(f"Metric '{name}' not registered")
        return metric
