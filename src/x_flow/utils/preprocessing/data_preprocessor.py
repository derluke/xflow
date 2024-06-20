from abc import ABC, abstractmethod
from copy import deepcopy

from ..data import Data


class DataPreprocessor(ABC):
    def fit(self, df: Data) -> "DataPreprocessor":
        return self._fit(df)

    def transform(self, df: Data) -> Data:
        df = deepcopy(df)
        return self._transform(df)

    def fit_transform(self, df: Data) -> Data:
        df = deepcopy(df)
        return self._fit(df)._transform(df)

    @abstractmethod
    def _fit(self, df: Data) -> "DataPreprocessor":
        ...

    @abstractmethod
    def _transform(self, df: Data) -> Data:
        ...


class Identity(DataPreprocessor):
    def _fit(self, df: Data) -> "Identity":
        return self

    def _transform(self, df: Data) -> Data:
        return df
