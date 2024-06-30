"""
This is a boilerplate pipeline 'deploy'
generated using Kedro 0.19.3.
"""

from typing import Any

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.settings import DYNAMIC_PIPELINES_MAPPING

from .nodes import deploy_models


def create_pipeline(**kwargs: Any) -> Pipeline:
    nodes = [
        node(
            func=deploy_models,
            inputs={
                "token": "params:credentials.datarobot.api_token",
                "endpoint": "params:credentials.datarobot.endpoint",
                "default_prediction_server_id": "params:credentials.datarobot.default_prediction_server_id",
                "best_models": "measure.best_models",
            },
            outputs="deployments",
        )
    ]

    deploy_template = pipeline(nodes)

    namespace = "deploy"
    pipes = []
    for variant in DYNAMIC_PIPELINES_MAPPING["experiment"]:
        modular_pipeline = pipeline(
            pipe=deploy_template,
            parameters={
                "params:credentials.datarobot.endpoint": "params:credentials.datarobot.endpoint",
                "params:credentials.datarobot.api_token": "params:credentials.datarobot.api_token",
                "params:credentials.datarobot.default_prediction_server_id": "params:credentials.datarobot.default_prediction_server_id",
            },
            inputs={"measure.best_models": f"measure.{variant}.best_models"},
            namespace=f"{namespace}.{variant}",
            tags=[variant, namespace],
        )
        pipes.append(modular_pipeline)
    return sum(pipes, Pipeline([]))
