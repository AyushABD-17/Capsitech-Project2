import pandas as pd
from sklearn.datasets import fetch_california_housing
from typing import Tuple, Union

import requests    
import io

def load_from_csv(path, target_col):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def load_from_api(url: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y




def load_dataset(source: str, **kwargs):
  
    if source == "csv":
        return load_from_csv(kwargs["path"], kwargs["target_col"])
    
    if source == "api":
        return load_from_api(kwargs["url"], kwargs["target_col"])
    
    raise ValueError(f"Unsupported source: {source}")
