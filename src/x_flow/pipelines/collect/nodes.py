"""
This is a boilerplate pipeline 'collect'
generated using Kedro 0.19.6
"""

from typing import Callable, Tuple

import pandas as pd



def create_leaderboard(backtest_results: dict[str, Callable[[None], pd.DataFrame]],
                       holdout_results: dict[str, Callable[[None], pd.DataFrame]],
                       external_holdout_results: dict[str, Callable[[None], pd.DataFrame]]
                       ) -> pd.DataFrame:
    
