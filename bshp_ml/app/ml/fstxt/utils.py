import re
import pandas as pd
from gensim.models import FastText
import numpy as np


def preprocess_text(text):
    """
    Функция для очистки текста
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    text = " ".join(list(filter(lambda x: not x.isnumeric(), text.split())))

    return text


def prepare_sentences(df, txt_cols):
    """
    Подготавливает предложения для обучения FastText
    """
    sentences = []

    for i in range(len(df)):
        combined_text = " ".join(
            [str(df[col].iloc[i]) for col in txt_cols if col in df.columns]
        )

        cleaned_text = preprocess_text(combined_text)
        if cleaned_text:
            words = cleaned_text.split()
            sentences.append(words)

    return sentences


def map_code_to_name():
    pass
    det_map = {name: num for num, name in enumerate(df[col].unique())}


def zero_below_nth_max(df, cols, treshold=0.8, n=10):
    """
    Занулить значения, меньшие чем n-е максимальное значение в каждой колонке
    """

    def row_max(row):
        row.nlargest(n).iloc[-1]

    df_result = df.copy()
    df_result[cols] = df_result[cols].where(df_result[cols] >= treshold, 0)

    return df_result
