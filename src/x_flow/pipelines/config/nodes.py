"""
This is a boilerplate pipeline 'config'
generated using Kedro 0.19.3
"""

from typing import Dict, List
import pandas as pd


def load_data(data: pd.DataFrame) -> List[Dict]:
    return data.to_dict(orient="records")
