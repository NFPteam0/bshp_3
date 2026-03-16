import pandas as pd
import numpy as np


def periodic_dates(
    date_col: pd.Series, dt: str = "month"
) -> tuple[pd.Series, pd.Series]:
    """
    date_col - номер месяца/дня/квартала
    """
    if dt == "month":
        div = 12
    elif dt == "day":
        div = 31
    else:
        div = 4
    _sin = np.sin((date_col * 2 * np.pi) / div)
    _cos = np.cos((date_col * 2 * np.pi) / div)
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
