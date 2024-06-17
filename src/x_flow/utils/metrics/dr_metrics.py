import logging
import re
from threading import Lock
from typing import Dict, Optional, Tuple

import datarobot as dr
import pandas as pd

from .metrics import Metric, get_holdout_metrics, get_otv_metrics

log = logging.getLogger(__name__)


datarobot_higher_is_better = {
    "Accuracy": True,
    "AUC": True,
    "Balanced Accuracy": True,
    "FVE Binomial": True,
    "Gini Norm": True,
    "Kolmogorov-Smirnov": True,
    "LogLoss": False,
    "Rate@Top5%": True,
    "Rate@Top10%": True,
    "Gamma Deviance": False,
    "FVE Gamma": True,
    "FVE Poisson": True,
    "FVE Tweedie": True,
    "MAD": False,
    "MAE": False,
    "MAPE": False,
    "Poisson Deviance": False,
    "R Squared": True,
    "RMSE": False,
    "RMSLE": False,
    "Tweedie Deviance": False,
}


def flatten_all_dr_metrics(metrics_dict: Dict) -> Dict[str, float]:
    flat_data = {}
    for metric_name, stats in metrics_dict.items():
        for key, value in stats.items():
            if key == "backtestingScores":
                for i, score in enumerate(value, start=1):
                    flat_data[f"{metric_name} {key}{i}"] = score
            else:
                flat_data[f"{metric_name} {key}"] = value

    # Create a DataFrame with a single row
    return flat_data


class DataRobotMetrics(Metric):
    def __init__(self):
        super().__init__(
            data_type="binary",
            name="datarobot_metrics",
            higher_is_better=datarobot_higher_is_better,
        )
        self.metrics_cache: Dict[Tuple[str, str], Dict] = {}
        self.locks: Dict[Tuple[str, str], Lock] = {}

    def get_lock(self, project_id, model_id):
        """Ensure that there is a unique lock for each (project_id, model_id) pair."""
        if (project_id, model_id) not in self.locks:
            self.locks[(project_id, model_id)] = Lock()
        return self.locks[(project_id, model_id)]

    def compute(  # noqa: PLR0913, PLR0911
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        extra_data: Optional[pd.DataFrame],
        experiment_config: Optional[dict],
        metric_config: Optional[dict],
        metadata: Optional[dict],
    ) -> Dict[str, Optional[float]]:
        if extra_data is None:
            log.warning("Model dataframe is not provided")
            return {"datarobot_metrics": None}
        if "model_id" not in extra_data.columns:
            log.warning("Model ID is not provided")
            return {"datarobot_metrics": None}
        if extra_data["model_id"].unique().shape[0] != 1:
            log.warning("Model ID is not unique")
            return {"datarobot_metrics": None}

        model_id = extra_data["model_id"].values[0]
        project_id = extra_data["project_id"].values[0]

        if metadata is None:
            log.warning("Metadata is not provided")
            return {"datarobot_metrics": None}
        if "data_subset" not in metadata:
            log.warning("Data subset is not provided")
            return {"datarobot_metrics": None}
        data_subset = metadata["data_subset"]

        lock = self.get_lock(project_id, model_id)
        with lock:
            if (project_id, model_id) not in self.metrics_cache:
                log.info(f"Fetching metrics for model {model_id}")
                model = dr.Model.get(project_id, model_id)  # type: ignore
                metrics = model.metrics
                self.metrics_cache[(project_id, model_id)] = metrics  # type: ignore
            else:
                log.info(f"Using cached metrics for model {model_id}")
                metrics = self.metrics_cache[(project_id, model_id)]

        if re.match(r"^\d+(\.0)?$", data_subset):
            # convert to int
            if data_subset.endswith(".0"):
                otv_index = int(data_subset[:-2])  # Drop the '.0' and convert
            else:
                otv_index = int(data_subset)  # Directly convert
            return get_otv_metrics(metrics, otv_index)  # type: ignore
        elif data_subset == "Holdout":
            # get the holdout metrics
            return get_holdout_metrics(metrics)  # type: ignore
        else:
            # return None
            return {}
