"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.19.3
"""

from datarobotx.idp.use_cases import get_or_create_use_case
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    binarize_data_node,
    get_or_create_dataset_from_df_with_lock,
    run_autopilot,
)


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
                func=binarize_data_node,
                inputs={
                    "input_data": "raw_data",
                    "target": "params:experiment_config.analyze_and_model.target",
                    "binarize_data_config": "params:experiment_config.binarize_data_config",
                },
                outputs=["data_binarized", "target_binarized"],
                name="binarize_data",
            ),
            node(
                func=lambda use_case_name, binarize_data_config: f"{use_case_name}{'_'+str(binarize_data_config) if binarize_data_config.get('binarize_operator') else ''}",
                inputs={
                    "use_case_name": "params:use_case_name",
                    "binarize_data_config": "params:experiment_config.binarize_data_config",
                },
                outputs="dataset_name",
                name="name_dataset",
            ),
            node(
                func=get_or_create_dataset_from_df_with_lock,
                inputs={
                    "token": "params:credentials.datarobot.api_token",
                    "endpoint": "params:credentials.datarobot.endpoint",
                    "use_case_id": "use_case_id",
                    "df": "data_binarized",
                    "name": "dataset_name",
                },
                outputs="experiment_dataset_id",
                name="get_or_create_dataset_from_df_with_lock",
            ),
            node(
                func=run_autopilot,
                inputs={
                    "token": "params:credentials.datarobot.api_token",
                    "endpoint": "params:credentials.datarobot.endpoint",
                    "target_name": "target_binarized",
                    "use_case_id": "use_case_id",
                    "dataset_id": "experiment_dataset_id",
                    "experiment_config": "params:experiment_config",
                },
                outputs="autopilot_run",
                name="get_or_create_autopilot_run",
            ),
        ],
    )
