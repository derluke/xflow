"""
This is a boilerplate pipeline 'config'
generated using Kedro 0.19.3
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_data(data: pd.DataFrame) -> List[Dict]:
    # replace any nan values with None
    # data_ = data.where(pd.notnull(data), None)
    data_ = data.replace({np.nan: None})
    records = data_.to_dict(orient="records")
    # replace any nan values with None
    return records


def decode_config(
    experiment_config: List[Dict], param_mapping: Dict[str, str]
) -> List[Dict]:
    # for each row in the config try to map the parameters
    # to the correct values
    log.info(f"Config: {experiment_config}")
    log.info(f"Decoding config with mapping: {param_mapping}")
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

    log.info(f"Decoded config: {decoded_configs}")
    return decoded_configs
