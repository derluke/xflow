"""
This is a boilerplate pipeline 'insights'
generated using Kedro 0.19.6
"""


import pandas as pd

from x_flow.utils.data import TrainingData

from datarobotx.idp.autopilot import get_or_create_autopilot_run
from datarobotx.idp.datasets import get_or_create_dataset_from_df


def train_test_prediction_project(
    endpoint: str,
    token: str,
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    usecase_id: str,
    name: str,
):
    # combine datasets by adding a new target: 0 = train, 1 = test
    training_df = training_data.copy()
    training_df["target"] = 0
    test_df = test_data.copy()
    test_df["target"] = 1
    df = pd.concat([training_df, test_df], axis=0).reset_index(drop=True)

    # create dataset
    dataset_id = get_or_create_dataset_from_df(
        endpoint=endpoint,
        token=token,
        data_frame=df,
        name=f"{name}_train_test_prediction_project",
        use_cases=usecase_id,
    )

    project_id = get_or_create_autopilot_run(
        endpoint=endpoint,
        token=token,
        dataset_id=dataset_id,
        name=f"{name}_train_test_prediction_project",
        use_case=usecase_id,
        analyze_and_model_config={
            "target": "target",
            "mode": "quick",
            "max_wait": 10000,
            "worker_count": -1,
        },
        advanced_options_config={
            "blend_best_models": False,
            "prepare_model_for_deployment": False,
            "min_secondary_validation_model_count": 0,
        },
    )

    return project_id
