"""
This is a boilerplate pipeline 'config'
generated using Kedro 0.19.3
"""

from copy import deepcopy
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        elif (key not in result or result[key] is None) and value is not None:
            result[key] = value
    return result


def load_data(data: pd.DataFrame) -> List[Dict]:
    # replace any nan values with None
    # data_ = data.where(pd.notnull(data), None)
    data_ = data.replace({np.nan: None})
    records = data_.to_dict(orient="records")
    # replace any nan values with None
    return records


def decode_config(
    experiment_config: List[Dict],
    param_mapping: Dict[str, str],
    global_parameters: Dict[str, str],
) -> List[Dict]:
    # for each row in the config try to map the parameters
    # to the correct values

    global_parameters = global_parameters.get("experiment_config", {})  # type: ignore
    log.info(f"Config: {json.dumps(experiment_config, indent=4, sort_keys=True)}")
    log.info(
        f"Decoding config with mapping: {json.dumps(param_mapping, indent=4, sort_keys=True)}"
    )
    log.info(
        f"Global parameters: {json.dumps(global_parameters, indent=4, sort_keys=True)}"
    )

    decoded_configs = []
    seen_experiment_names = set()

    for experiment_row in experiment_config:
        decoded_config = {}
        for experiment_key, experiment_value in experiment_row.items():
            if experiment_key == "experiment_name":
                if experiment_value in seen_experiment_names:
                    raise ValueError(
                        f"Duplicate experiment name detected: {experiment_value}"
                    )
                seen_experiment_names.add(experiment_value)
                decoded_config[experiment_key] = experiment_value
                continue
            if experiment_key in param_mapping:
                if param_mapping[experiment_key] not in decoded_config:
                    decoded_config[param_mapping[experiment_key]] = {}

                decoded_config[param_mapping[experiment_key]][
                    experiment_key
                ] = experiment_value
        decoded_configs.append(decoded_config)

    log.info(
        f"Decoded config pre merge: {json.dumps(decoded_configs, indent=4, sort_keys=True)}"
    )

    # deep merge globals into the decoded configs if not overridden
    merged_configs = []
    for decoded_config in decoded_configs:
        merged_config = merge_dicts(decoded_config, global_parameters)
        merged_configs.append(merged_config)
    log.info(
        f"Decoded config post merge: {json.dumps(merged_configs, indent=4, sort_keys=True)}"
    )

    def prune_none_values(d: Dict) -> Dict:
        if not isinstance(d, Dict):
            return d

        new_dict = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, Dict):
                pruned = prune_none_values(v)
                if pruned:
                    new_dict[k] = pruned
            else:
                new_dict[k] = v

        return new_dict

    merged_pruned_dicts = [prune_none_values(d) for d in merged_configs]
    log.info(
        f"Decoded config post prune: {json.dumps(merged_pruned_dicts, indent=4, sort_keys=True)}"
    )
    return merged_pruned_dicts
