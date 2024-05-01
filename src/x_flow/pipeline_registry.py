"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.pipelines.config.pipeline import create_pipeline as create_config_pipeline
from x_flow.pipelines.experiment.pipeline import (
    create_pipeline as create_experiment_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    config_pipeline = pipeline(
        create_config_pipeline(),
        namespace="config",
        inputs={"experiments": "experiments", "param_mapping": "param_mapping"},
    )

    experiments_pipeline = pipeline(
        create_experiment_pipeline(),
        namespace="dr_experiment",
        inputs={
            # "use_case_name": "use_case_name",
            "raw_data_train": "raw_data_train",
            "raw_data_test": "raw_data_test",
        },
        parameters={
            "params:credentials.datarobot.api_token": "params:credentials.datarobot.api_token",
            "params:credentials.datarobot.endpoint": "params:credentials.datarobot.endpoint",
            "params:use_case_name": "params:use_case_name",
        },
    )
    x_flow_experiments_pipeline = pipeline(
        create_experiment_pipeline(),
        namespace="experiment",
        inputs={
            # "use_case_name": "use_case_name",
            "raw_data_train": "raw_data_train",
            "raw_data_test": "raw_data_test",
        },
        parameters={
            "params:credentials.datarobot.api_token": "params:credentials.datarobot.api_token",
            "params:credentials.datarobot.endpoint": "params:credentials.datarobot.endpoint",
            "params:use_case_name": "params:use_case_name",
            # "params:experiment_config": "params:experiment_config",
            # "params:experiment_name": "params:experiment_name",
        },
        # outputs="autopilot_run",
    )

    pipelines = {
        "__default__": config_pipeline
        + experiments_pipeline
        + x_flow_experiments_pipeline,
        "config": config_pipeline,
        "experiment": experiments_pipeline,
    }

    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    return pipelines
