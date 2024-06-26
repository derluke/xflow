"""
This is a boilerplate pipeline 'deploy'
generated using Kedro 0.19.3.
"""

from typing import Any

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import deploy_models


def create_pipeline(**kwargs: Any) -> Pipeline:
    return pipeline(
        [
            node(
                func=deploy_models,
                inputs={
                    "token": "params:credentials.datarobot.api_token",
                    "endpoint": "params:credentials.datarobot.endpoint",
                    "default_prediction_server_id": "params:credentials.datarobot.default_prediction_server_id",
                    "best_models": "best_models",
                },
                outputs="deployments",
            )
        ],
    )
