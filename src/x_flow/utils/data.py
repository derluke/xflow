from collections import OrderedDict
from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
from typing import Any, Optional, Union

from kedro.io import AbstractDataset
import pandas as pd


@dataclass(kw_only=True)
class Data:
    df: pd.DataFrame
    columns: Optional[list[str]] = None
    row_index: Optional[list[int]] = None
    date_partition_column: Optional[Union[str, list[datetime.datetime]]] = None
    partition_column: Optional[str] = None
    date_column: Optional[str] = None
    date_format: str = None

    # @staticmethod
    # def from_file(filepath: Path, **kwargs) -> "Data":
    #     df = pd.read_csv(filepath)
    #     return Data(df=df, **kwargs)

    # @staticmethod
    # def from_df(df: pd.DataFrame, **kwargs) -> "Data":
    #     return Data(df=df, **kwargs)

    @property
    def rendered_df(self) -> pd.DataFrame:
        df = self.df
        if self.columns is not None:
            columns = set(self.columns)
            if self.date_column:
                columns.add(self.date_column)
            if hasattr(self, "target_column") and self.target_column:
                columns.add(self.target_column)
            if self.partition_column:
                columns.add(self.partition_column)
            if isinstance(self.date_partition_column, str):
                columns.add(self.date_partition_column)
            df = df[list(columns)]
        if self.row_index is not None:
            df = df.iloc[self.row_index]
        if self.date_column and df[self.date_column].dtype != "datetime64[ns]":
            df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format)
        return df.reset_index(drop=True)

    def get_date_partitions(self) -> dict[str, pd.DataFrame]:
        df = self.rendered_df
        if self.date_column is None:
            raise ValueError("date column is not set")
        start_date = df[self.date_column].min()
        end_date = df[self.date_column].max()
        if self.date_partition_column is None:
            return {"__all__": df}
        elif isinstance(self.date_partition_column, str):
            return {
                str(group): group_df.reset_index(drop=True)
                for group, group_df in df.groupby(self.date_partition_column)
            }
        elif isinstance(self.date_partition_column, list):
            partition_dates = self.date_partition_column
            return {
                start_date: df[
                    (df[self.date_column] >= start_date) & (df[self.date_column] <= end_date)
                ].reset_index(drop=True)
                for start_date, end_date in zip(
                    [start_date] + partition_dates, partition_dates + [end_date]
                )
            }

    def get_partitions(self) -> OrderedDict[str, pd.DataFrame]:
        df = self.rendered_df
        if self.partition_column is not None:
            return OrderedDict(
                {
                    str(group): group_df.reset_index(drop=True)
                    for group, group_df in df.groupby(self.partition_column)
                }
            )
        else:
            return OrderedDict({"__all_data__": df})


@dataclass(kw_only=True)
class TrainingData(Data):
    target_column: str


@dataclass(kw_only=True)
class ValidationData(TrainingData): ...


@dataclass(kw_only=True)
class ValidationPredictionData(ValidationData): ...


@dataclass(kw_only=True)
class PredictionData(Data): ...


class_map: dict[str, type] = {
    "Data": Data,
    "TrainingData": TrainingData,
    "ValidationData": ValidationData,
    "ValidationPredictionData": ValidationPredictionData,
    "PredictionData": PredictionData,
}


class XFlowDataset(AbstractDataset[Data, Data]):
    def __init__(self, filepath: str, **kwargs: Any):
        self._filepath = filepath
        self.kwargs = kwargs
        load_args = kwargs.get("load_args", {})
        load_class = load_args.get("load_class", "Data")
        if load_class not in class_map:
            raise ValueError(f"load_class must be one of {list(class_map.keys())}")
        self.load_class = load_class

    def _load(self, **kwargs: Any) -> Data:
        df = pd.read_csv(Path(self._filepath))
        with open(Path(self._filepath).parent / "metadata.json") as f:
            metadata = json.load(f)
        instance: Data = class_map[self.load_class](df=df, **metadata)
        return instance

    def _save(self, data: Data) -> None:
        Path(self._filepath).mkdir(parents=True, exist_ok=True)
        data.df.to_csv(Path(self._filepath) / "data.csv", index=False)
        metadata = asdict(data)
        metadata.pop("df")
        with open(Path(self._filepath) / "metadata.json", "w") as f:
            json.dump(metadata, f, default=str)

    def _describe(self) -> dict[str, Any]:
        return dict(
            filepath=self._filepath,
            **self.kwargs,
        )
