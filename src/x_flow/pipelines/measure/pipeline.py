"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_metrics,
                inputs={
                    "experiment_config": "params:experiment_config",
                    # "target_binarized": "target_binarized",
                    "metric_config": "params:metric_config",
                    "metrics": "params:metrics",
                    "predictions": "holdouts",
                },
                outputs="holdout_metrics",
                name="calculate_holdout_metrics",
            ),
            node(
                func=calculate_metrics,
                inputs={
                    "experiment_config": "params:experiment_config",
                    # "target_binarized": "target_binarized",
                    "metric_config": "params:metric_config",
                    "metrics": "params:metrics",
                    "predictions": "backtests",
                },
                outputs="backtest_metrics",
                name="calculate_backtest_metrics",
            ),
            node(
                func=calculate_metrics,
                inputs={
                    "experiment_config": "params:experiment_config",
                    # "target_binarized": "target_binarized",
                    "metric_config": "params:metric_config",
                    "metrics": "params:metrics",
                    "predictions": "external_holdout",
                },
                outputs="external_holdout_metrics",
                name="calculate_external_holdout_metrics",
            ),
        ]
    )
