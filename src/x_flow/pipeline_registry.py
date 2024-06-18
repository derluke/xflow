"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.pipelines.config.pipeline import create_pipeline as create_config_pipeline
from x_flow.pipelines.experiment.pipeline import (
    create_pipeline as create_experiment_pipeline,
)
from x_flow.pipelines.measure.pipeline import create_pipeline as create_measure_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    config_pipeline = pipeline(
        create_config_pipeline(),
        namespace="config",
        inputs={"experiments": "experiments", "param_mapping": "param_mapping"},
        parameters={"params:global_params": "params:experiment"},
    )

    x_flow_experiments_pipeline = pipeline(
        create_experiment_pipeline(),
        namespace="experiment",
        inputs={
            "raw_data_train": "raw_data_train",
            "raw_data_test": "raw_data_test",
        },
        parameters={
            "params:credentials.datarobot.api_token": "params:credentials.datarobot.api_token",
            "params:credentials.datarobot.endpoint": "params:credentials.datarobot.endpoint",
            "params:use_case_name": "params:use_case_name",
        },
        outputs={
            "backtests": "experiment.backtests",
            "holdouts": "experiment.holdouts",
            "external_holdout": "experiment.external_holdout",
        },
    )

    x_flow_measure_pipeline = pipeline(
        create_measure_pipeline(),
        namespace="measure",
        inputs={
            # "target_binarized": "experiment.target_binarized",
            "holdouts": "experiment.holdouts",
            "backtests": "experiment.backtests",
            "external_holdout": "experiment.external_holdout",
        },
        outputs={
            "holdout_metrics": "measure.holdout_metrics",
            "backtest_metrics": "measure.backtest_metrics",
            "external_holdout_metrics": "measure.external_holdout_metrics",
        },
        parameters={
            # "params:experiment_config": "params:experiment.experiment_config",
            # "params:metrics": "params:experiment.metrics",
            # "params:metric_config": "params:experiment.metric_config",
        },
    )

    pipelines = {
        "__default__": config_pipeline
        + x_flow_experiments_pipeline
        + x_flow_measure_pipeline,
        "config": config_pipeline,
    }

    return pipelines
