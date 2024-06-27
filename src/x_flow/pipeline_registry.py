"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline
from kedro.framework.project import find_pipelines


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns
    -------
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    return pipelines
