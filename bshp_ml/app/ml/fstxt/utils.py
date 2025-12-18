import re
import pandas as pd
from gensim.models import FastText
import numpy as np

RE_CHARS = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]")
RE_SPACES = re.compile(r"\s+")


def preprocess_text(s: pd.Series) -> pd.Series:
    """
    Функция для очистки текста
    """
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(RE_CHARS, " ", regex=True)
    s = s.str.replace(RE_SPACES, " ", regex=True).str.strip()

    return s


def prepare_sentences(df: pd.DataFrame, txt_cols) -> list[list[str]]:
    """
    Подготавливает предложения для обучения FastText
    """
    df_txt = df[txt_cols].astype(str)
    df_txt = df_txt.agg(" ".join, axis=1)

    cleaned = preprocess_text(df_txt)

    sentences = cleaned[cleaned != ""].str.split().to_list()

    return sentences


def map_code_to_name():
    pass
    # det_map = {name: num for num, name in enumerate(df[col].unique())}


def zero_below_nth_max(df, cols, treshold=0.8, n=10):
    """
    Занулить значения, меньшие чем n-е максимальное значение в каждой колонке
    """

    def row_max(row):
        row.nlargest(n).iloc[-1]

    df_result = df.copy()
    df_result[cols] = df_result[cols].where(df_result[cols] >= treshold, 0)

    return df_result
