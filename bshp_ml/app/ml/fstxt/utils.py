import re
import pandas as pd
from gensim.models import FastText
import numpy as np

RE_CHARS = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]")
RE_SPACES = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"[^a-zA-Zа-яА-ЯёЁ\s]")
RE_SMALL = re.compile(r"\b\w{1-2}\b")


def preprocess_text(s: pd.Series) -> pd.Series:
    """
    Функция для очистки текста
    """
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(RE_CHARS, " ", regex=True)
    s = s.str.replace(RE_SPACES, " ", regex=True).str.strip()
    s = s.str.replace(RE_SMALL, " ", regex=True)

    return s


def prepare_sentences(df: pd.DataFrame, txt_cols) -> list[list[str]]:
    """
    Подготавливает предложения для обучения FastText
    """
    df_txt = df[txt_cols].astype(str)

    HIGH_IMP = [
        "article_name",
        "payment_purpose",
        "payment_purpose_returned",
        "analytic",
        "analytic2",
        "analytic3",
    ]
    for col in HIGH_IMP:
        if col in df_txt.columns:
            if col == "payment_purpose" or col == "payment_purpose_returned":
                df_txt[col] = df_txt[col].str.replace(RE_NUMBERS, " ", regex=True)
            df_txt[col] = df_txt[col] + " "

    # df_txt = df_txt[HIGH_IMP].agg(" ".join, axis=1)
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
