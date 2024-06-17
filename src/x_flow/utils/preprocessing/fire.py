from datarobotx.idp.autopilot import get_or_create_autopilot_run
from datarobotx.idp.common.hashing import get_hash
from datarobotx.idp.datasets import get_or_create_dataset_from_df

from ..data import Data
from .data_preprocessor import DataPreprocessor
from .fire_implementation import FIRE as FireHelper


class FIRE(DataPreprocessor):
    def __init__(self, endpoint: str, token: str, **kwargs):
        self._endpoint = endpoint
        self._token = token
        self._fire_kwargs = kwargs

    def _fit(self, df: Data):
        dataset_id = get_or_create_dataset_from_df(
            endpoint=self._endpoint,
            token=self._token,
            data_frame=df.rendered_df.to_pandas(),
            name="fire_dataset",
        )
        fire_token = get_hash(**self._fire_kwargs)
        project_id = get_or_create_autopilot_run(
            endpoint=self._endpoint,
            token=self._token,
            dataset_id=dataset_id,
            name=f"fire_project [{fire_token}]",
            analyze_and_model_config={
                "target": df.target_column,
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

        fire = FireHelper.get(project_id=project_id)  # type: ignore
        fire.main_feature_reduction(
            **self._fire_kwargs,
        )

        top_featurelist_name = fire.get_top_model().featurelist_name
        top_featurelist = fire.get_featurelist_by_name(top_featurelist_name)  # type: ignore
        features = top_featurelist.features  # type: ignore
        self._features = features
        return self

    def _transform(self, df: Data) -> Data:
        """helper function: select features from FIRE project"""
        df.columns = self._features
        return df
