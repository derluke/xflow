"""
This is a boilerplate pipeline 'collect'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_leaderboard, collect_deployments


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=collect_deployments,
                inputs="deployments_combined",
                outputs="collected_deployments",
                name="collect_deployments",
            ),
            node(
                func=create_leaderboard,
                inputs={
                    "backtest_results": "backtest_metrics_combined",
                    "holdout_results": "holdout_metrics_combined",
                    "external_holdout_results": "external_holdout_metrics_combined",
                },
                outputs="leaderboard",
                name="create_leaderboard",
            ),
        ],
    )
