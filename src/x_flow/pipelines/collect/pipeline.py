"""
This is a boilerplate pipeline 'collect'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.settings import DYNAMIC_PIPELINES_MAPPING

from .nodes import create_leaderboard, collect_deployments


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda *args: args,
                inputs=[
                    f"{variant}.backtest_metrics_grouped"
                    for variant in DYNAMIC_PIPELINES_MAPPING["measure"]
                ],
                outputs="backtest_metrics_combined",
                name="combine_backtest_metrics",
            ),
            node(
                func=lambda *args: args,
                inputs=[
                    f"{variant}.holdout_metrics_grouped"
                    for variant in DYNAMIC_PIPELINES_MAPPING["measure"]
                ],
                outputs="holdout_metrics_combined",
                name="combine_holdout_metrics",
            ),
            node(
                func=lambda *args: args,
                inputs=[
                    f"{variant}.external_holdout_metrics_grouped"
                    for variant in DYNAMIC_PIPELINES_MAPPING["measure"]
                ],
                outputs="external_holdout_metrics_combined",
                name="combine_external_holdout_metrics",
            ),
            node(
                func=lambda *args: args,
                inputs=[
                    f"{variant}.deployments" for variant in DYNAMIC_PIPELINES_MAPPING["experiment"]
                ],
                outputs="deployments_combined",
                name="combine_deployments",
            ),
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
        inputs=dict(
            {
                f"{variant}.{data_subset}_metrics_grouped": f"measure.{variant}.{data_subset}_metrics_grouped"
                for variant in DYNAMIC_PIPELINES_MAPPING["measure"]
                for data_subset in ["backtest", "holdout", "external_holdout"]
            },
            **{
                f"{variant}.deployments": f"deploy.{variant}.deployments"
                for variant in DYNAMIC_PIPELINES_MAPPING["experiment"]
            },
        ),
        namespace="collect",
    )
