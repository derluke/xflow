"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.pipelines.config.pipeline import create_pipeline as create_config_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    config_pipeline = pipeline(
        create_config_pipeline(),
        namespace="config",
        inputs={"experiments": "experiments"},
    )

    pipelines = {
        "__default__": config_pipeline,
        "config": config_pipeline,
    }

    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    return pipelines
