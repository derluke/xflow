"""
This is a boilerplate pipeline 'dataprep'
generated using Kedro 0.19.5.
"""

from typing import Any, Tuple

from x_flow.utils.data import Data
from x_flow.utils.preprocessing.binary_transformer import BinarizeData
from x_flow.utils.preprocessing.data_preprocessor import DataPreprocessor, Identity
from x_flow.utils.preprocessing.fire import FIRE


def preprocessing_fit_transform(
    data: Data, *transformations: DataPreprocessor
) -> Tuple[Data, bool]:
    for transformation in transformations:
        data = transformation.fit_transform(data)
    return data, True


def preprocessing_transform(
    data: Data,
    fit_complete: bool,
    *transformations: DataPreprocessor,
) -> Data:
    if not fit_complete:
        raise ValueError("Fit must be completed before transform")
    for transformation in transformations:
        data = transformation.transform(data)
    return data


def register_binarize_preprocessor(binarize_data_config: dict[str, Any]) -> DataPreprocessor:
    if binarize_data_config is None:
        transformer = Identity()
    else:
        transformer = BinarizeData(**binarize_data_config)
    return transformer


def register_fire_preprocessor(fire_config: dict[str, Any]) -> DataPreprocessor:
    if fire_config is None:
        transformer = Identity()
    else:
        transformer = FIRE(**fire_config)
    return transformer
