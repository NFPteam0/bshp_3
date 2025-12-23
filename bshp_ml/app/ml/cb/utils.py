import gc
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score


def eval_model(true_labels, predictions) -> tuple:
    ac = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    print(f"Accuracy: {ac:.4f}, F1: {f1:.4f}")
    return ac, f1


def make_all_data(df: pd.DataFrame, y: str):
    """
    Создает копию df длинной len(df[y].unique()), т.е.
    общий срез датасета, где каждый объект искомого
    класса - уникальный
    """
    # .reset_index()??
    all_data = df.groupby(df[f"{y}_norm"]).first().copy()
    all_data.reset_index(inplace=True, drop=True)
    all_data[f"{y}_norm"] = all_data.index
    return all_data


def encode_cat(col: pd.Series, encoder: dict = None):
    if encoder:
        return col.map(encoder), encoder
    else:
        encoder = {name: num for num, name in enumerate(col.unique())}
        return col.map(encoder), encoder


def decode_cat(col: pd.Series, decoder: dict = None):
    if decoder:
        return col.map(decoder), decoder
    else:
        decoder = {name: num for num, name in enumerate(col.unique())}
        return col.map(decoder), decoder


def get_none_data_row(self, parameters):
    row = {}
    for col in parameters["x_columns"] + parameters["y_columns"]:
        if col in parameters["float_columns"]:
            row[col] = 0
        elif col in parameters["str_columns"]:
            row[col] = "None"
        elif col in parameters["bool_columns"]:
            row[col] = False
        else:
            row[col] = None

    return pd.DataFrame([row])


def get_y_map(df: pd.DataFrame, y: list[str], mmap: dict | None = None) -> pd.DataFrame:
    # TODO: записать в декодер
    rows = []
    if mmap is None:
        mmap = {name: num for num, name in enumerate(df[y].unique())}
    for item in df[y].unique():
        ymap = dict()
        ymap["value"] = item
        ymap["code"] = df[[y, f"{y}_name"]].first()

        ymap["norm"] = df[y].map(mmap)
        rows.append(ymap)
    return pd.DataFrame(rows)
