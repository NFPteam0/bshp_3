from datetime import datetime
import logging
import os
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from settings import USE_DETAILED_LOG

logging.getLogger("bshp_data_processing_logger")
logger = logging.getLogger(__name__)


class CBDataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict
        self.x_columns = self.parameters["x_columns"]
        self.y_columns = self.parameters["y_columns"]
        self.additional_columns = self.parameters["additional_columns"]
        self.columns_to_encode = self.parameters["columns_to_encode"]
        self.form_encode_dict = False

        self.df = pd.DataFrame(columns=["name", "code1c", "code_norm"])
        self.encode_dict = {}
        self.decode_dict = {}

    def fit(self, X, y=None):
        self.set_mapping(X, y)

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
            return self.decode_dict[field][value]

    def _get_encoded_field(self, value, field):
        if not value:
            return -1
        else:
            return self.encode_dict[field][value]

    def _get_month(self, date_value: datetime):
        return date_value.month

    def _get_year(self, date_value: datetime):
        return date_value.year

    def set_mapping(
        self, X: pd.DataFrame, y: str, mmap: dict | None = None
    ) -> pd.DataFrame:
        rows = []
        if mmap is None:
            mmap = {name: num for num, name in enumerate(X[y].unique())}
        for item in X[y].unique():
            ymap = dict()
            ymap["name"] = X[
                [y, f"{y}_name"]
            ].first()  # либо _name, либо _name_pred, если есть
            ymap["code1c"] = item
            ymap["code_norm"] = X[y].map(mmap)
            rows.append(ymap)
        self.df = pd.DataFrame(rows)

    def save(self, folder, name, encoder):
        path_to_model = os.path.join(folder, name)
        if not os.path.isdir(path_to_model):
            os.makedirs(path_to_model)

        with open(os.path.join(path_to_model, "encoder.pkl"), "wb") as fp:
            pickle.dump(encoder, fp)
