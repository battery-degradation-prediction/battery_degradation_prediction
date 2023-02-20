"""load data module"""
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load csv file into pandas dataframe

    Parameters
    ----------
    path : str
        path to the data

    Returns
    -------
    df : pd.DataFrame
        A dataframe containing battery data
    """
    df = pd.read_csv(path)
    return df
