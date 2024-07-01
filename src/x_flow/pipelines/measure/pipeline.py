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
        "backtest",
        "holdout",
        "external_holdout",
    ]

    nodes = []
    for grouped in [False, True]:
        for subset in data_subsets:
            input_dict = {
                "experiment_config": "params:experiment_config",
                "metric_config": "params:measure_config.metric_config",
                "metrics": "params:measure_config.metrics",
                "prediction_data": f"{subset}",
            }
            if grouped:
                input_dict["best_models"] = "_best_models"
            nodes.append(
                node(
                    func=calculate_metrics,
                    inputs=input_dict,
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
            outputs="_best_models",
        )
    )

    nodes.append(
        node(
            name="best_models",
            func=lambda best_models: best_models,
            inputs={
                "best_models": "_best_models",
            },
            outputs="best_models",
        )
    )

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
                "holdout": f"experiment.{variant}.holdout",
                "backtest": f"experiment.{variant}.backtest",
                "external_holdout": f"experiment.{variant}.external_holdout",
            },
            namespace=f"{namespace}.{variant}",
            tags=[variant, namespace],
        )
        pipes.append(modular_pipeline)
    return sum(pipes, Pipeline([]))
