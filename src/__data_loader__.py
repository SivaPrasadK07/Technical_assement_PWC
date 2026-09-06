import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
        file_path (str): The path to the CSV file.
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    return pd.read_csv(file_path)

def get_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
     Get a head sample of rows from a pandas DataFrame.
        df (pd.DataFrame): The input DataFrame.
        n (int): The number of rows to sample.
        pd.DataFrame: The sampled data as a pandas DataFrame.
    """
    return df.head(n)