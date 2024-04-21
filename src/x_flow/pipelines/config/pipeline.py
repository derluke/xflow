"""
This is a boilerplate pipeline 'config'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import load_data, decode_config


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=load_data, inputs="experiments", outputs="experiments_records"),
            node(
                func=decode_config,
                inputs={
                    "experiment_config": "experiments_records",
                    "param_mapping": "param_mapping",
                },
                outputs="decoded_config",
                name="decode_config",
            ),
        ],
    )
