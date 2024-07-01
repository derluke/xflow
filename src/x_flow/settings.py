"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html.
"""

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
# from x_flow.hooks import ProjectHooks
from copy import deepcopy
from typing import Any

from kedro.config import OmegaConfigLoader  # noqa: E402
import omegaconf
import pandas as pd
import yaml

from x_flow.pipelines.config.nodes import decode_config, load_data
from x_flow.utils.common.checkpoint_hooks import CheckpointHooks

from datarobotx.idp.common.credentials_hooks import CredentialsHooks

# Hooks are executed in a Last-In-First-Out (LIFO) order.
# HOOKS = (ProjectHooks(),)
HOOKS = (CredentialsHooks(), CheckpointHooks())
APP_CLASS = "controller.app.XFlowApp"

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.


def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.

    Args:
        dict1 (dict): The first dictionary to merge.
        dict2 (dict): The second dictionary to merge.

    Returns:
        dict: The merged dictionary.
    """
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if (
            isinstance(value, omegaconf.dictconfig.DictConfig)
            and key in result
            and isinstance(result[key], omegaconf.dictconfig.DictConfig)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


experiments = pd.read_csv("include/x_flow/config/experiments.csv")
global_params = yaml.safe_load(open("include/x_flow/config/param_mapping.yml"))


experiments_dict = load_data(experiments)
decoded_experiments = decode_config(experiments_dict, global_params, {})

multi_pipelines = ["experiment", "measure", "dataprep"]
overrides: dict[str, Any] = {}  # {"experiment": {"project": "${..project}"}}
for multi_pipeline in multi_pipelines:
    overrides[multi_pipeline] = {}
    for experiment in decoded_experiments:
        experiment_name = experiment["experiment_name"]
        # print(experiment_name)
        override_dict = {
            "_overrides": experiment,
            f"{multi_pipeline}_config": f"${{merge:${{...{multi_pipeline}_config}},${{._overrides}}}}",
        }
        overrides[multi_pipeline][experiment_name] = override_dict

# write only if contents have changed
with open("conf/base/parameters_overrides.yml") as f:
    existing_content = f.read()
if existing_content != yaml.dump(overrides, sort_keys=False):
    with open("conf/base/parameters_overrides.yml", "w") as f:
        yaml.dump(overrides, f, sort_keys=False)


CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {
        "merge": merge_dicts,
    },
}


DYNAMIC_PIPELINES_MAPPING = {
    "experiment": list(overrides["experiment"].keys()),
    "measure": list(overrides["experiment"].keys()),
}
# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
