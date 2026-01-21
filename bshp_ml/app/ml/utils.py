import pandas as pd
import numpy as np


def pereodic_dates(date_col: pd.Series):
    _sin = np.sin((date_col.dt.month * 2 * np.pi) / 12)
    _cos = np.cos((date_col.dt.month * 2 * np.pi) / 12)
    return (_sin, _cos)


def move_column(df: pd.DataFrame, col: str):
    """Двигает колонку в самую правую позицию
    (сделано для того, чтобы текстовой модели
    было легче предсказать след. слово)"""
    if col in df.columns:
        cols_ = df.columns.to_list()
        cols_.remove(col)
        cols_.append(col)
        df = df[cols_]
    return df
