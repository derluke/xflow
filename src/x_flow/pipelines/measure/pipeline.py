"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.19.3.
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.settings import DYNAMIC_PIPELINES_MAPPING

from .nodes import calculate_metrics, get_best_models


def create_pipeline(**kwargs) -> Pipeline:
    data_subsets = [
        "backtests",
        "holdouts",
        "external_holdout",
    ]

    nodes = []
    for grouped in [False, True]:
        for subset in data_subsets:
            nodes.append(
                node(
                    func=calculate_metrics,
                    inputs={
                        "experiment_config": "params:experiment_config",
                        "metric_config": "params:measure_config.metric_config",
                        "metrics": "params:measure_config.metrics",
                        "prediction_data": f"{subset}",
                    },
                    outputs=f"{subset}_metrics" + "_grouped" * grouped,
                    name=f"calculate_{subset}_metrics" + "_grouped" * grouped,
                )
            )
    nodes.append(
        node(
            name="get_best_models",
            func=get_best_models,
            inputs={
                "metrics_by_partition": "external_holdout_metrics",
                "experiment_config": "params:experiment_config",
            },
            outputs="best_models",
        )
    )

    # nodes = [
    #     node(
    #         func=calculate_metrics,
    #         inputs={
    #             "experiment_config": "params:experiment_config",
    #             "metric_config": "params:measure_config.metric_config",
    #             "metrics": "params:measure_config.metrics",
    #             "prediction_data": "holdouts",
    #         },
    #         outputs="holdout_metrics",
    #         name="calculate_holdout_metrics",
    #     ),
    #     node(
    #         func=calculate_metrics,
    #         inputs={
    #             "experiment_config": "params:experiment_config",
    #             "metric_config": "params:measure_config.metric_config",
    #             "metrics": "params:measure_config.metrics",
    #             "prediction_data": "backtests",
    #         },
    #         outputs="backtest_metrics",
    #         name="calculate_backtest_metrics",
    #     ),
    #     node(
    #         func=calculate_metrics,
    #         inputs={
    #             "experiment_config": "params:experiment_config",
    #             "metric_config": "params:measure_config.metric_config",
    #             "metrics": "params:measure_config.metrics",
    #             "prediction_data": "external_holdout",
    #         },
    #         outputs="external_holdout_metrics",
    #         name="calculate_external_holdout_metrics",
    #     ),
    #     node(
    #         name="get_best_models",
    #         func=get_best_models,
    #         inputs={
    #             "metrics_by_partition": "external_holdout_metrics",
    #             "experiment_config": "params:experiment_config",
    #         },
    #         outputs="best_models",
    #     ),
    #     node(
    #         func=calculate_metrics,
    #         inputs={
    #             "experiment_config": "params:experiment_config",
    #             "metric_config": "params:measure_config.metric_config",
    #             "metrics": "params:measure_config.metrics",
    #             "prediction_data": "holdouts",
    #             "best_models": "best_models",
    #         },
    #         outputs="holdout_metrics_grouped",
    #         name="calculate_holdout_metrics_grouped",
    #     ),
    #     node(
    #         func=calculate_metrics,
    #         inputs={
    #             "experiment_config": "params:experiment_config",
    #             "metric_config": "params:measure_config.metric_config",
    #             "metrics": "params:measure_config.metrics",
    #             "prediction_data": "backtests",
    #             "best_models": "best_models",
    #         },
    #         outputs="backtest_metrics_grouped",
    #         name="calculate_backtest_metrics_grouped",
    #     ),
    #     node(
    #         func=calculate_metrics,
    #         inputs={
    #             "experiment_config": "params:experiment_config",
    #             "metric_config": "params:measure_config.metric_config",
    #             "metrics": "params:measure_config.metrics",
    #             "prediction_data": "external_holdout",
    #             "best_models": "best_models",
    #         },
    #         outputs="external_holdout_metrics_grouped",
    #         name="calculate_external_holdout_metrics_grouped",
    #     ),
    # ]

    measure_template = pipeline(nodes)
    namespace = "measure"
    variants = DYNAMIC_PIPELINES_MAPPING["experiment"]
    pipes = []
    for variant in variants:
        modular_pipeline = pipeline(
            pipe=measure_template,
            parameters={
                "params:experiment_config": f"params:experiment.{variant}.experiment_config"
            },
            inputs={
                "holdouts": f"{variant}.holdouts",
                "backtests": f"{variant}.backtests",
                "external_holdout": f"{variant}.external_holdout",
            },
            namespace=f"{namespace}.{variant}",
            tags=[variant, namespace],
        )
        pipes.append(modular_pipeline)
    return sum(pipes, Pipeline([]))
