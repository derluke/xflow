"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3.
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_metrics, get_best_models


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
                    "prediction_data": "holdouts",
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
                    "prediction_data": "backtests",
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
                    "prediction_data": "external_holdout",
                },
                outputs="external_holdout_metrics",
                name="calculate_external_holdout_metrics",
            ),
            node(
                name="get_best_models",
                func=get_best_models,
                inputs={
                    "metrics_by_partition": "external_holdout_metrics",
                    "experiment_config": "params:experiment_config",
                },
                outputs="best_models",
            ),
            node(
                func=calculate_metrics,
                inputs={
                    "experiment_config": "params:experiment_config",
                    # "target_binarized": "target_binarized",
                    "metric_config": "params:metric_config",
                    "metrics": "params:metrics",
                    "prediction_data": "holdouts",
                    "best_models": "best_models",
                },
                outputs="holdout_metrics_grouped",
                name="calculate_holdout_metrics_grouped",
            ),
            node(
                func=calculate_metrics,
                inputs={
                    "experiment_config": "params:experiment_config",
                    # "target_binarized": "target_binarized",
                    "metric_config": "params:metric_config",
                    "metrics": "params:metrics",
                    "prediction_data": "backtests",
                    "best_models": "best_models",
                },
                outputs="backtest_metrics_grouped",
                name="calculate_backtest_metrics_grouped",
            ),
            node(
                func=calculate_metrics,
                inputs={
                    "experiment_config": "params:experiment_config",
                    # "target_binarized": "target_binarized",
                    "metric_config": "params:metric_config",
                    "metrics": "params:metrics",
                    "prediction_data": "external_holdout",
                    "best_models": "best_models",
                },
                outputs="external_holdout_metrics_grouped",
                name="calculate_external_holdout_metrics_grouped",
            ),
        ]
    )
