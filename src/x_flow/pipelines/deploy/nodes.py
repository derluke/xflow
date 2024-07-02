"""
This is a boilerplate pipeline 'deploy'
generated using Kedro 0.19.3.
"""

# freeze models and deploy

# output: list of deployment_ids and nice bit of code which can be used to schedule predictions from outside of DR
# weekly run, take last 5 days of data

from typing import Any

import pandas as pd

from datarobotx.idp.deployments import get_or_create_deployment_from_registered_model_version
from datarobotx.idp.registered_model_versions import (
    get_or_create_registered_leaderboard_model_version,
)


def deploy_models(
    endpoint: str, token: str, best_models: pd.DataFrame, default_prediction_server_id: str
):
    best_models_dict = best_models.to_dict(orient="records")
    deployments: dict[str, Any] = {}
    for model in best_models_dict:
        # call deployment API here, save the id of each deployed model
        project_id = model["project_id"]
        model_id = model["model_id"]
        experiment = model["experiment_name"]
        partition = model["partition"]
        if experiment not in deployments:
            deployments[experiment] = []
        registered_model_name = f"{experiment}_{project_id}_{model_id}"
        registered_model_version_id = get_or_create_registered_leaderboard_model_version(
            endpoint=endpoint,
            token=token,
            model_id=model_id,
            registered_model_name=registered_model_name,
        )
        deployment_id = get_or_create_deployment_from_registered_model_version(
            endpoint=endpoint,
            token=token,
            registered_model_version_id=registered_model_version_id,
            label=f"Deployment of {experiment}_{project_id}_{model_id}",
            default_prediction_server_id=default_prediction_server_id,
        )
        deployments[experiment].append(
            {
                "deployment_id": deployment_id,
                "project_id": project_id,
                "model_id": model_id,
                "partition": partition,
            }
        )
    return deployments
