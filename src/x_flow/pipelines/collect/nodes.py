"""
This is a boilerplate pipeline 'collect'
generated using Kedro 0.19.6
"""

from typing import Tuple

import pandas as pd

import pandas as pd


def reshape_dataframe(df, data_fold=""):
    # Create a new dictionary to hold the reshaped data
    reshaped_data = {}

    # Add experiment_name (should be the same for all rows)
    reshaped_data["experiment_name"] = df["experiment_name"].iloc[0]

    # Add project_id and model_id (should be the same for all rows)
    reshaped_data["project_id"] = df["project_id"].iloc[0]
    reshaped_data["model_id"] = df["model_id"].iloc[0]

    # # Add rank (should be the same for all rows)
    # reshaped_data['rank'] = df['rank'].iloc[0]

    # Loop through each row and add metrics for each data subset
    for _, row in df.iterrows():
        subset = f"data_subset_{row['data_subset']}"
        for col in df.columns:
            if col not in ["experiment_name", "partition", "project_id", "model_id", "data_subset"]:
                reshaped_data[f"{data_fold}_{subset}_{col}"] = row[col]

    # Create a new DataFrame from the reshaped data
    return pd.DataFrame([reshaped_data])


def create_leaderboard(
    backtest_results: list[pd.DataFrame],
    holdout_results: list[pd.DataFrame],
    external_holdout_results: list[pd.DataFrame],
) -> pd.DataFrame:
    backtest_dfs = []
    holdout_dfs = []
    external_holdout_dfs = []
    for df in backtest_results:
        transformed_df = reshape_dataframe(df, data_fold="backtests")
        backtest_dfs.append(transformed_df)
    for df in holdout_results:
        transformed_df = reshape_dataframe(df, data_fold="holdout")
        holdout_dfs.append(transformed_df)
    for df in external_holdout_results:
        transformed_df = reshape_dataframe(df, data_fold="external_holdout")
        external_holdout_dfs.append(transformed_df)

    join_cols = ["experiment_name", "project_id", "model_id"]
    result = (
        pd.concat(external_holdout_dfs)
        .set_index(join_cols)
        .join(pd.concat(holdout_dfs).set_index(join_cols), on=join_cols, rsuffix="holdout_")
        .join(pd.concat(backtest_dfs).set_index(join_cols), on=join_cols, rsuffix="backtests_")
    )
    return result.dropna(axis=1, how="all")


def collect_deployments(deployments):
    all_deployments = []
    for deployment_dict in deployments:
        experiment_name = list(deployment_dict.keys())[0]
        deployment_records = deployment_dict[experiment_name]
        for record in deployment_records:
            record["experiment_name"] = experiment_name
            all_deployments.append(record)

    df_deployments = pd.DataFrame(all_deployments)
    return df_deployments
