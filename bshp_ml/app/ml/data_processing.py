import logging
import asyncio
import zipfile
import os
import pandas as pd
import json
from datetime import datetime, timezone
from pydantic import TypeAdapter, ValidationError

# from pydantic_core import InitErrorDetails
# from typing import List
# from collections import defaultdict
# from .cb.utils import get_y_map

from sklearn.base import BaseEstimator, TransformerMixin

# import asyncio
# import uuid
import gc

# from tasks import task_manager
from settings import TEMP_FOLDER, USE_DETAILED_LOG, DB_URL
from schemas.models import DataRow
# from db import db_processor


logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class Checker(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if USE_DETAILED_LOG:
            logger.info("Start checking data")
        if X.empty:
            raise ValueError(
                "Fitting dataset is empty. Load more data or change filter."
            )
        if USE_DETAILED_LOG:
            logger.info("Checking data. Done. Shape, %s", str(X.shape))
        return X


class DataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict
        self.x_columns = self.parameters["x_columns"]
        self.y_columns = self.parameters["y_columns"]
        self.additional_columns = self.parameters["additional_columns"]
        self.columns_to_encode = self.parameters["columns_to_encode"]
        self.form_encode_dict = False

        self.encode_dict = {}
        self.decode_dict = {}

    def fit(self, X, y=None):
        if USE_DETAILED_LOG:
            logger.info("Start encoding data, shape: %s", str(X.shape))
            logger.info(
                "Form encode dict for columns: %s",
                ", ".join([col for col in X.columns if col in self.columns_to_encode]),
            )
        if self.form_encode_dict:
            for col in self.columns_to_encode:
                uniq = list(X[col].unique())

                uniq = [el for el in uniq if el]

                enc_dict = dict(zip(uniq, range(len(uniq))))
                self.encode_dict[col] = enc_dict

        return self

    def transform(self, X):
        if USE_DETAILED_LOG:
            logger.info("Encoding, shape: %s", str(X.shape))
        for col in self.columns_to_encode:
            X[col] = X[col].apply(lambda x: self._get_encoded_field(x, col))

        X = X[self.additional_columns + self.x_columns + self.y_columns]
        if USE_DETAILED_LOG:
            logger.info("Encoding data. Done. Shape: %s", str(X.shape))
        return X

    def inverse_transform(self, X):
        if USE_DETAILED_LOG:
            logger.info("Start decoding data. Shape: %s", str(X.shape))
        self.decode_dict = {}
        if self.form_encode_dict:
            for col in self.encode_dict:
                d = {v: k for k, v in self.encode_dict[col].items()}
                self.decode_dict[col] = d

        for col in self.columns_to_encode:
            X[col] = X[col].apply(lambda x: self._get_decoded_field(x, col))
        if USE_DETAILED_LOG:
            logger.info("Decoding data. Done. Shape: %s", str(X.shape))
        return X

    def _get_decoded_field(self, value, field):
        if value == -1:
            return ""
        else:
            return self.decode_dict[field].get(value, -1)

    def _get_encoded_field(self, value, field):
        if not value:
            return -1
        else:
            return self.encode_dict[field].get(value, -1)


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict

        self.x_columns = self.parameters["x_columns"]
        self.y_columns = self.parameters["y_columns"]
        self.additional_columns = self.parameters["additional_columns"]
        self.str_columns = self.parameters["str_columns"]
        self.float_columns = self.parameters["float_columns"]
        self.date_columns = self.parameters["date_columns"]

    def fit(self, X, y=None):
        return self

    def transform(self, X_df):
        X = X_df.copy()
        if USE_DETAILED_LOG:
            logger.info("Feature adding")

        for col in self.date_columns:
            if X[col].dtype in ["string", "object"] and col != "uploading_date":
                # X[col] = X[col].apply(str_to_date)
                X[col] = pd.to_datetime(
                    X[col],
                    format="mixed",
                    errors="coerce",
                    dayfirst=True,
                ).fillna(pd.Timestamp("1970-01-01"))

        X["document_year"] = X["date"].apply(self._get_year)
        X["document_month"] = X["date"].apply(self._get_month)

        # X = X[self.additional_columns + self.x_columns + self.y_columns].copy()
        if USE_DETAILED_LOG:
            logger.info("Adding features. Done. Shape: %s", str(X.shape))
        return X

    def _get_month(self, date_value: datetime):
        return date_value.month

    def _get_year(self, date_value: datetime):
        return date_value.year


class NanProcessor(BaseEstimator, TransformerMixin):
    """Transformer for working with nan values (deletes nan rows, columns, fills 0 to na values)"""

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict

        self.str_columns = self.parameters["str_columns"]
        self.float_columns = self.parameters["float_columns"]

    def fit(self, X, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Process nan values: removes all nan rows and columns, fills 0 instead single nan values
        :param x: data before nan processing
        :return: data after na  processing
        """
        x = x.copy()
        if USE_DETAILED_LOG:
            logger.info("Start processing Nan values")
        self.str_columns = [col for col in self.str_columns if col in x.columns]
        self.float_columns = [col for col in self.float_columns if col in x.columns]

        x[self.str_columns] = x[self.str_columns].fillna("")
        x[self.float_columns] = x[self.float_columns].fillna(0)
        x.loc[x["year"] == "", "year"] = "0"
        if USE_DETAILED_LOG:
            logger.info("Processing Nan values. Done")
        return x


class Shuffler(BaseEstimator, TransformerMixin):
    """
    Transformer class to shuffle data rows
    """

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict

    def fit(self, X, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if USE_DETAILED_LOG:
            logger.info("Start data shuffling")
        result = x.sample(frac=1).reset_index(drop=True).copy()
        if USE_DETAILED_LOG:
            logger.info("Data shuffling. Done")
        return result


def str_to_date(value):
    try:
        return datetime.strptime(value, r"%d.%m.%Y %H:%M:%S")
    except:  # noqa: E722
        return datetime(1970, 1, 1, 0, 0, 0)
