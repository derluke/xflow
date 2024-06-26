"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.pipelines.config.pipeline import create_pipeline as create_config_pipeline
from x_flow.pipelines.experiment.pipeline import (
    create_pipeline as create_experiment_pipeline,
)
from x_flow.pipelines.measure.pipeline import create_pipeline as create_measure_pipeline
from x_flow.pipelines.deploy.pipeline import create_pipeline as create_deploy_pipeline
from x_flow.pipelines.collect.pipeline import create_pipeline as create_collect_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns
    -------
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
            "holdout_metrics_grouped": "measure.holdout_metrics_grouped",
            "backtest_metrics_grouped": "measure.backtest_metrics_grouped",
            "external_holdout_metrics_grouped": "measure.external_holdout_metrics_grouped",
            "best_models": "measure.best_models",
        },
        parameters={
            # "params:experiment_config": "params:experiment.experiment_config",
            # "params:metrics": "params:experiment.metrics",
            # "params:metric_config": "params:experiment.metric_config",
        },
    )

    deploy_pipeline = pipeline(
        create_deploy_pipeline(),
        namespace="deploy",
        inputs={"best_models": "measure.best_models"},
        parameters={
            "params:credentials.datarobot.api_token": "params:credentials.datarobot.api_token",
            "params:credentials.datarobot.endpoint": "params:credentials.datarobot.endpoint",
            "params:credentials.datarobot.default_prediction_server_id": "params:credentials.datarobot.default_prediction_server_id",
        },
        outputs="deployments",
    )

    collect_pipeline = pipeline(
        create_collect_pipeline(),
        namespace="collect",
        # inputs={
        #     "deployments_combined",
        #     "backtest_metrics_combined",
        #     "holdout_metrics_combined",
        #     "external_holdout_metrics_combined",
        # },
        # parameters={"deployments_combined": "collect.deployments_combined"},
    )

    pipelines = {
        "__default__": config_pipeline
        + x_flow_experiments_pipeline
        + x_flow_measure_pipeline
        + deploy_pipeline
        + collect_pipeline,
        # + collect_pipeline,
        "config": config_pipeline,
        "collect": collect_pipeline,
    }

    return pipelines
