import logging
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score
from ..pipelines.experiment.nodes import Operator

log = logging.getLogger(__name__)


def generalized_f1(
    actuals: pd.Series,
    predictions: pd.Series,
    extra_data: Optional[pd.DataFrame],
    metric_config: Optional[dict],
    experiment_config: Optional[dict],
) -> float:
    """
    Calculate generalized F1 metric.

    Args:
        actuals: Actual target values.
        predictions: Predicted target values.
        extra_data: Additional data.
        metric_config: Metric configuration.

    Returns:
        float: Generalized AUC metric value.
    """
    # calculate generalized AUC
    # get binarize options:
    if experiment_config is None:
        raise ValueError("experiment_config is required")

    binarize_operator = experiment_config.get("binarize_data", {}).get(
        "binarize_operator"
    )
    binarize_threshold = experiment_config.get("binarize_data", {}).get(
        "binarize_threshold"
    )
    # log.warning(f"binarize_operator: {binarize_operator}")
    # log.warning(f"binarize_threshold: {binarize_threshold}")

    if binarize_operator and binarize_threshold:
        binarized_predictions = predictions
    else:
        if not metric_config or "binarize_data_config" not in metric_config:
            raise ValueError("metric_config is required")

        binarize_operator = metric_config["binarize_data_config"].get("operator", None)
        binarize_threshold = metric_config["binarize_data_config"].get(
            "threshold", None
        )
        # log.warning(f"{metric_config}")
        # log.warning(f"binarize_operator: {binarize_operator}")
        # log.warning(f"binarize_threshold: {binarize_threshold}")
        op_fun = Operator(operator=binarize_operator).apply_operation(
            binarize_threshold
        )
        binarized_predictions = op_fun(predictions)

    # log.info(f"binarized_predictions: {binarized_predictions}")
    # ensure actuals are boolean
    actuals = actuals.astype(bool)
    binarized_predictions = binarized_predictions.astype(bool)

    # calculate generalized F1
    return float(f1_score(actuals, binarized_predictions))
