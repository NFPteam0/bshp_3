import gc
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool


# def make_full(df: pd.DataFrame, col, i=0, j=-1):
#     """
#     fills dataset for each batch (df[i:j]) to have every class presented in all_classes
#     """
#     classes_df = df[i:j].groupby(df[col]).first().copy()
#     classes_df.reset_index(drop=True, inplace=True)

#     if not all_data.loc[all_data.index.difference(classes_df.index)].empty:
#         Xy = pd.concat(
#             [
#                 df[i:j],
#                 all_data.loc[all_data.index.difference(classes_df[f"{col}_norm"])],
#             ],
#             ignore_index=True,
#         )
#     else:
#         Xy = df[i:j]
#     print(len(all_data))

#     return Xy


# def get_batch_pool(df: pd.DataFrame, batch_size: int = 1000):
#     i = 0
#     j = 0
#     while i < len(df):
#         j = min(i + batch_size, len(df))
#         Xy = make_full(df, col, i, j)
#         y_batch = Xy[f"{col}_norm"]
#         X_batch = Xy.drop([col for col in y_columns if col in Xy.columns], axis=1)
#         yield Pool(
#             X_batch,
#             label=y_batch,
#             cat_features=cats,
#         )
#         del Xy, y_batch, X_batch
#         gc.collect()
#         i = j


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
