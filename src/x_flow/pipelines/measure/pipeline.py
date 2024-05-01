"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

# from .nodes import select_candidate_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=select_candidate_models,
            #     inputs={
            #         "projects": "projects",
            #         "metric": "params:experiment_config.metric",
            #         "metric_config": "params:experiment_config.metric_config",
            #         "experiment_config": "params:experiment_config",
            #     },
            #     outputs="model_candidates",
            #     name="select_candidate_models",
            # )
        ]
    )
