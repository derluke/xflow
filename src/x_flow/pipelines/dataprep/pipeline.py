"""
This is a boilerplate pipeline 'dataprep'
generated using Kedro 0.19.5.
"""

from typing import Any

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.settings import DYNAMIC_PIPELINES_MAPPING
from x_flow.utils.data import TrainingData, ValidationData

from .nodes import (
    preprocessing_fit_transform,
    preprocessing_transform,
    register_binarize_preprocessor,
    register_fire_preprocessor,
)


def create_pipeline(**kwargs: Any) -> Pipeline:
    nodes = [
        node(
            name="load_raw_data",
            func=lambda df, target_column, partition_column, date_column, date_format: TrainingData(
                df=df,
                target_column=target_column,
                partition_column=partition_column,
                date_column=date_column,
                date_format=date_format,
            ),
            inputs={
                "df": "raw_data_train",
                "target_column": "params:experiment_config.analyze_and_model.target",
                "partition_column": "params:dataprep_config.group_data.partition_column",
                "date_column": "params:experiment_config.datetime_partitioning.datetime_partition_column",
                "date_format": "params:dataprep_config.date_format",
            },
            outputs="data_train",
        ),
        node(
            name="register_binarize_preprocessor",
            func=register_binarize_preprocessor,
            inputs={
                "binarize_data_config": "params:dataprep_config.binarize_data",
            },
            outputs="binarize_data_transformer",
        ),
        node(
            name="register_fire_preprocessor",
            func=register_fire_preprocessor,
            inputs={
                "fire_config": "params:dataprep_config.fire_config",
            },
            outputs="fire_transformer",
        ),
        node(
            name="apply_preprocessing",
            func=preprocessing_fit_transform,
            inputs=[
                "data_train",
                "binarize_data_transformer",
                "fire_transformer",
            ],
            outputs=["data_train_transformed", "preprocessor_fit_complete"],
        ),
        node(
            name="load_raw_data_test",
            func=lambda df,
            target_column,
            partition_column,
            date_column,
            date_format: ValidationData(
                df=df,
                target_column=target_column,
                partition_column=partition_column,
                date_column=date_column,
                date_format=date_format,
            ),
            inputs={
                "df": "raw_data_test",
                "target_column": "params:experiment_config.analyze_and_model.target",
                "partition_column": "params:dataprep_config.group_data.partition_column",
                "date_column": "params:experiment_config.datetime_partitioning.datetime_partition_column",
                "date_format": "params:dataprep_config.date_format",
            },
            outputs="data_test",
        ),
        node(
            func=preprocessing_transform,
            inputs=[
                "data_test",
                "preprocessor_fit_complete",
                "binarize_data_transformer",
                "fire_transformer",
            ],
            outputs="data_test_transformed",
            name="apply_preprocessing_test",
        ),
    ]
    experiment_template = pipeline(nodes)

    namespace = "dataprep"
    pipes = []
    for variant in DYNAMIC_PIPELINES_MAPPING["experiment"]:
        modular_pipeline = pipeline(
            pipe=experiment_template,
            inputs={"raw_data_train", "raw_data_test"},
            parameters={
                "params:experiment_config.analyze_and_model.target": f"params:experiment.{variant}.experiment_config.analyze_and_model.target",
                "params:experiment_config.datetime_partitioning.datetime_partition_column": f"params:experiment.{variant}.experiment_config.datetime_partitioning.datetime_partition_column",
            },
            namespace=f"{namespace}.{variant}",
            tags=[variant, namespace],
        )
        pipes.append(modular_pipeline)
    return sum(pipes, Pipeline([]))
