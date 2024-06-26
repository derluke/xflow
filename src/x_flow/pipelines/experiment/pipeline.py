try:
    from datarobot import UseCase  # type: ignore  # noqa: F401

    from datarobotx.idp.use_cases import get_or_create_use_case  # type: ignore
except ImportError:

    def get_or_create_use_case(*args, **kwags):  # type: ignore
        return "not_supported"


from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from x_flow.utils.data import TrainingData, ValidationData

from .nodes import (
    calculate_backtests,
    get_backtest_predictions,
    get_external_predictions,
    get_or_create_dataset_from_df_with_lock,
    get_or_create_use_case_with_lock,
    preprocessing_fit_transform,
    preprocessing_transform,
    register_binarize_preprocessor,
    register_fire_preprocessor,
    run_autopilot,
    unlock_holdouts,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_or_create_use_case_with_lock,
                inputs={
                    "token": "params:credentials.datarobot.api_token",
                    "endpoint": "params:credentials.datarobot.endpoint",
                    "name": "params:use_case_name",
                },
                outputs="use_case_id",
                name="get_or_create_use_case",
            ),
            node(
                name="load_raw_data",
                func=lambda df,
                target_column,
                partition_column,
                date_column,
                date_format: TrainingData(
                    df=df,
                    target_column=target_column,
                    partition_column=partition_column,
                    date_column=date_column,
                    date_format=date_format,
                ),
                inputs={
                    "df": "raw_data_train",
                    "target_column": "params:experiment_config.analyze_and_model.target",
                    "partition_column": "params:experiment_config.partition_column",
                    "date_column": "params:experiment_config.datetime_partitioning.datetime_partition_column",
                    "date_format": "params:experiment_config.date_format",
                },
                outputs="data_train",
            ),
            node(
                name="register_binarize_preprocessor",
                func=register_binarize_preprocessor,
                inputs={
                    "binarize_data_config": "params:experiment_config.binarize_data_config",
                },
                outputs="binarize_data_transformer",
            ),
            node(
                name="register_fire_preprocessor",
                func=register_fire_preprocessor,
                inputs={
                    "fire_config": "params:experiment_config.fire_config",
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
                outputs="data_train_transformed",
            ),
            node(
                name="name_dataset",
                func=lambda use_case_name,
                binarize_data_config: f"{use_case_name}{'_'+str(binarize_data_config) if binarize_data_config and binarize_data_config.get('binarize_operator') else ''}",
                inputs={
                    "use_case_name": "params:use_case_name",
                    "binarize_data_config": "params:experiment_config.binarize_data_config",
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
                tags=["checkpoint"],
            ),
            node(
                func=lambda project_dict, df, backtests_completed: backtests_completed
                and get_backtest_predictions(project_dict, df),
                inputs={
                    "project_dict": "project_dict",
                    "df": "data_train_transformed",
                    "backtests_completed": "backtests_completed",
                },
                outputs="backtests",
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
                outputs="holdouts",
                name="get_holdout_predictions",
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
                    "partition_column": "params:experiment_config.partition_column",
                    "date_column": "params:experiment_config.datetime_partitioning.datetime_partition_column",
                    "date_format": "params:experiment_config.date_format",
                },
                outputs="data_test",
            ),
            node(
                func=preprocessing_transform,
                inputs=["data_test", "binarize_data_transformer", "fire_transformer"],
                outputs="data_test_transformed",
                name="apply_preprocessing_test",
            ),
            node(
                func=get_external_predictions,
                inputs={
                    "project_dict": "project_dict",
                    "external_holdout": "data_test_transformed",
                },
                outputs="external_holdout",
                name="get_external_holdout_predictions",
            ),
        ],
    )
