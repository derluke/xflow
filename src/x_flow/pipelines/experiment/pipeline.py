from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.settings import DYNAMIC_PIPELINES_MAPPING

from .nodes import (
    calculate_backtests,
    get_backtest_predictions,
    get_external_predictions,
    get_or_create_dataset_from_df_with_lock,
    get_or_create_use_case_with_lock,
    run_autopilot,
    unlock_holdouts,
)


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        node(
            func=get_or_create_use_case_with_lock,
            inputs={
                "token": "params:credentials.datarobot.api_token",
                "endpoint": "params:credentials.datarobot.endpoint",
                "name": "params:experiment_config.use_case_name",
            },
            outputs="use_case_id",
            name="get_or_create_use_case",
        ),
        node(
            name="name_dataset",
            func=lambda use_case_name,
            binarize_data_config: f"{use_case_name}{'_'+str(binarize_data_config) if binarize_data_config and binarize_data_config.get('binarize_operator') else ''}",
            inputs={
                "use_case_name": "params:experiment_config.use_case_name",
                "binarize_data_config": "params:binarize_data",
            },
            outputs="dataset_name",
        ),
        node(
            func=get_or_create_dataset_from_df_with_lock,
            inputs={
                "token": "params:credentials.datarobot.api_token",
                "endpoint": "params:credentials.datarobot.endpoint",
                "use_case_id": "use_case_id",
                "df": "data_train_transformed",
                "name": "dataset_name",
            },
            outputs="experiment_dataset_dict",
            name="get_or_create_dataset_from_df_with_lock",
        ),
        node(
            func=run_autopilot,
            inputs={
                "token": "params:credentials.datarobot.api_token",
                "endpoint": "params:credentials.datarobot.endpoint",
                "df": "data_train_transformed",
                "use_case_id": "use_case_id",
                "dataset_dict": "experiment_dataset_dict",
                "experiment_config": "params:experiment_config",
            },
            outputs="project_dict",
            name="get_or_create_project_dict",
        ),
        node(
            func=unlock_holdouts,
            inputs={
                "project_dict": "project_dict",
            },
            outputs="holdouts_unlocked",
            name="unlock_holdouts",
        ),
        node(
            func=calculate_backtests,
            inputs={
                "project_dict": "project_dict",
                "holdouts_unlocked": "holdouts_unlocked",
            },
            outputs="backtests_completed",
            name="calculate_backtests",
        ),
        node(
            func=lambda project_dict, df, backtests_completed: backtests_completed
            and get_backtest_predictions(project_dict, df),
            inputs={
                "project_dict": "project_dict",
                "df": "data_train_transformed",
                "backtests_completed": "backtests_completed",
            },
            outputs="backtest",
            # tags=["checkpoint"],
            name="get_backtest_predictions",
        ),
        node(
            func=lambda project_dict, df, backtests_completed: backtests_completed
            and get_backtest_predictions(project_dict, df, data_subset="holdout"),
            inputs={
                "project_dict": "project_dict",
                "df": "data_train_transformed",
                "backtests_completed": "backtests_completed",
            },
            outputs="holdout",
            # tags=["checkpoint"],
            name="get_holdout_predictions",
        ),
        node(
            func=get_external_predictions,
            inputs={
                "project_dict": "project_dict",
                "external_holdout": "data_test_transformed",
            },
            outputs="external_holdout",
            # tags=["checkpoint"],
            name="get_external_holdout_predictions",
        ),
    ]

    experiment_template = pipeline(nodes)

    namespace = "experiment"
    pipes = []
    for variant in DYNAMIC_PIPELINES_MAPPING[namespace]:
        modular_pipeline = pipeline(
            pipe=experiment_template,
            inputs={
                "data_train_transformed": f"dataprep.{variant}.data_train_transformed",
                "data_test_transformed": f"dataprep.{variant}.data_test_transformed",
            },
            parameters={
                "params:credentials.datarobot.endpoint": "params:credentials.datarobot.endpoint",
                "params:credentials.datarobot.api_token": "params:credentials.datarobot.api_token",
                "binarize_data": f"dataprep.{variant}.dataprep_config.binarize_data",
            },
            namespace=f"{namespace}.{variant}",
            tags=[variant, namespace],
        )
        pipes.append(modular_pipeline)
    return sum(pipes, Pipeline([]))
