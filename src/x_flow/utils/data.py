import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import polars as pl


@dataclass(kw_only=True)
class Data:
    df: pl.DataFrame
    columns: Optional[list[str]] = None
    row_index: Optional[list[int]] = None
    date_partition_column: Optional[Union[str, list[datetime.datetime]]] = None
    partition_column: Optional[str] = None
    date_column: Optional[str] = None
    date_format: str = "%Y-%m-%d"
    target_column: str

    @staticmethod
    def load(load_path: Path, target_column: str, **kwargs) -> "Data":
        df = pl.read_csv(
            load_path,
        )
        return Data(df=df, target_column=target_column, **kwargs)

    @property
    def rendered_df(self) -> pl.DataFrame:
        df = self.df
        if self.columns is not None and len(self.columns):
            columns = self.columns
            if self.date_column is not None:
                columns = [self.date_column] + columns
            if self.target_column is not None:
                columns = [self.target_column] + columns
            if self.partition_column is not None:
                columns = [self.partition_column] + columns
            if self.date_partition_column is not None and isinstance(
                self.date_partition_column, str
            ):
                columns = [self.date_partition_column] + columns
            columns = set(columns)
            df = df.select(columns)
        if self.row_index is not None and len(self.row_index):
            df = df[self.row_index]
        if (
            self.date_column is not None
            and not df.dtypes[df.columns.index(self.date_column)] == pl.Datetime
        ):
            df = df.with_columns(
                pl.col(self.date_column).str.to_datetime(self.date_format)
            )
        return df

    def get_date_partitions(self):
        df = self.rendered_df

        if self.date_column is None:
            raise ValueError("date column is not set")
        start_date = df[self.date_column].min()
        end_date = df[self.date_column].max()
        if self.date_partition_column is None:
            return {"__all__": df}

        elif isinstance(self.date_partition_column, str):
            return {
                group: group_df
                for group, group_df in df.groupby([self.date_partition_column])
            }

        elif isinstance(self.date_partition_column, list) and isinstance(
            self.date_partition_column[0], datetime.datetime
        ):
            partition_dates = self.date_partition_column
            return {
                start_date: df.filter(
                    df[self.date_column].is_between(start_date, end_date)
                )
                for start_date, end_date in zip(
                    [start_date] + partition_dates, partition_dates + [end_date]
                )
            }

    def get_partitions(self):
        df = self.rendered_df
        if self.partition_column is not None:
            return {
                group: df_group
                for group, df_group in df.groupby([self.partition_column])
            }
        else:
            return {"__all__": df}


@dataclass(kw_only=True)
class TrainingData(Data): ...


class ExternalHoldoutData(Data): ...
