"""
This is a boilerplate pipeline 'insights'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import train_test_prediction_project
from datarobotx.idp.use_cases import get_or_create_use_case


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_or_create_use_case,
                inputs={
                    "token": "params:credentials.datarobot.api_token",
                    "endpoint": "params:credentials.datarobot.endpoint",
                    "name": "params:use_case_name",
                },
                outputs="use_case_id",
                name="get_or_create_use_case",
            ),
            node(
                func=train_test_prediction_project,
                inputs={
                    "token": "params:credentials.datarobot.api_token",
                    "endpoint": "params:credentials.datarobot.endpoint",
                    "training_data": "raw_data_train",
                    "test_data": "raw_data_test",
                    "usecase_id": "use_case_id",
                    "name": "params:use_case_name",
                },
                outputs="train_test_project_id",
                name="train_test_prediction_project",
            ),
        ],
        inputs={
            "raw_data_train": "raw_data_train",
            "raw_data_test": "raw_data_test",
        },
        namespace="insights",
        parameters={
            "params:credentials.datarobot.api_token": "params:credentials.datarobot.api_token",
            "params:credentials.datarobot.endpoint": "params:credentials.datarobot.endpoint",
            "params:use_case_name": "params:use_case_name",
        },
    )
