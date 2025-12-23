import gc
import json
import logging
import os
import pickle
import random
import shutil
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import UTC, datetime
from enum import Enum
from typing import Optional
from sklearn import set_config
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from ml.cb.classifier import (
    CatBoostClassifier,
    Pool,
    sum_models,
    to_classifier,
)
from schemas.models import ModelStatuses, ModelTypes
from settings import (
    DATASET_BATCH_LENGTH,
    MODEL_FOLDER,
    QUANTIZE,
    THREAD_COUNT,
    USE_DETAILED_LOG,
    USED_RAM_LIMIT,
)
from sklearn.pipeline import Pipeline
from .utils import get_none_data_row
import copy
from ml.data_processing import (
    Checker,
    DataEncoder,
    FeatureAdder,
    NanProcessor,
)
from db import db_processor
from tasks.__init__ import Reader
from ..models import Model
from .classifier import CatBoostModel, CbCallBack
from .utils import decode_cat, encode_cat, eval_model, make_all_data
from .data_processing import CBDataEncoder

logging.getLogger("bshp_data_processing_logger")
logger = logging.getLogger(__name__)
SEED = 42


class CatBoostModelEmbeddings(CatBoostModel):
    """Класс модели catboost с пониманием эмбеддингов"""

    model_type = ModelTypes.catboost_txt

    field_models: dict[str, CatBoostClassifier] | None = None
    categorical: list[str] | None = (
        None  # названия категориальных колонок, need to be normalized
    )
    # cat_idxs: list[int] | None

    def __init__(self, base_name):
        super().__init__(base_name)
        self.need_to_encode = False
        self.categorical = [
            "moving_type",
            "company_inn",
            "company_kpp",  # TODO: убрать
            "base_document_kind",
            "contractor_inn",
            "contractor_kind",
            "company_account_number",
            "contractor_account_number",
            "cash_flow_item_code",
            "year",
            "cash_flow_details_code",
        ]
        self.fsttxt_columns = ["cash_flow_item_name", "cash_flow_details_name", "year"]
        self.float_columns.extend([f"prob_{y}" for y in self.fsttxt_columns])
        self.categorical.extend([f"pred_{y}" for y in self.fsttxt_columns])
        self.str_columns.extend(
            [f"pred_{y}" for y in self.fsttxt_columns] + ["payment_purpose"]
        )
        self.x_columns.extend(
            [f"pred_{y}" for y in self.fsttxt_columns] + ["payment_purpose"]
        )
        self.columns_to_encode = self.categorical

        self.date_columns = [
            "date",
            "base_document_date",
            "article_document_date",
            # "uploading_date",
        ]
        self.parameters = {
            "x_columns": self.x_columns,
            "y_columns": self.y_columns,
            "str_columns": self.str_columns,
            "float_columns": self.float_columns,
            "bool_columns": self.bool_columns,
            "additional_columns": self.additional_columns,
            "columns_to_encode": self.columns_to_encode,
            "categorical": self.categorical,
            "date_columns": self.date_columns,
        }

    def train(
        self,
        params: dict,
        y: str,
        to_drop: list[str],
        cat_idxs: list[int],
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        test_pool: Pool,
        all_data: pd.DataFrame,
    ) -> CatBoostClassifier:
        """Trains one catboost model.
        <b>NOTE: before fitting add columns from fasttext model</b>"""
        model = None
        model_new = None

        # test_pool = self.get_test_pool(
        #     df_test=df_test, y=y, to_drop=to_drop, cat_idxs=cat_idxs
        # )

        for n, batch in enumerate(
            self.get_batch_pool(
                df=df_train,
                y=y,
                to_drop=to_drop,
                cat_idxs=cat_idxs,
                batch_size=25_000,
                all_data=all_data,
            )
        ):
            if USE_DETAILED_LOG:
                logger.info(f"BATCH {n} of total {len(df_train) // 25_000 + 1}")
            if model is None:
                model = CatBoostClassifier(**params)
                batch.set_baseline(np.zeros(shape=(batch.num_row(), len(all_data))))
                test_pool.set_baseline(
                    np.zeros(shape=(test_pool.num_row(), len(all_data)))
                )

                model.fit(X=batch, eval_set=test_pool)
            else:
                model_new = CatBoostClassifier(**params)
                base = model.predict(batch, prediction_type="RawFormulaVal")
                base[np.isneginf(base)] = -1
                batch.set_baseline(base)

                test_base = model.predict(test_pool, prediction_type="RawFormulaVal")
                test_base[np.isneginf(test_base)] = -1
                test_pool.set_baseline(test_base)

                model_new.fit(X=batch, eval_set=test_pool)
                model = sum_models(
                    [model, model_new], ctr_merge_policy="IntersectingCountersAverage"
                )  # TODO: weights = [1/2] * 2?
            del model_new
            gc.collect()
        return model

    def gridsearch(
        self,
        all_classes: pd.DataFrame,
        df_train,
        df_test,
        cat_idxs: list[int],
        test_pool: Pool,
        y: str,
        to_drop: list[str],
        all_data: pd.DataFrame,
        lr=0.01,
        trees=20,
    ):
        """
        Traines few models and choses best based on a given catboost parametrs
        """
        param_grid = {
            "learning_rate": [
                lr,
                #   0.002,
                # 0.004,
            ],
            "depth": [6],
            "iterations": [
                trees,
                trees * 2,
                max(int(trees * 0.7), 1),
                #   300,
                #   400,
            ],
            "l2_leaf_reg": [4],
        }

        best_score = 0
        best_model = None
        best_params = None

        # Перебираем все комбинации параметров
        for lr in param_grid["learning_rate"]:
            for depth in param_grid["depth"]:
                for iterations in param_grid["iterations"]:
                    if USE_DETAILED_LOG:
                        logging.info(
                            f"\n=== Testing: lr={lr}, depth={depth}, iterations={iterations} ==="
                        )

                    params = {
                        "task_type": "CPU",
                        "iterations": iterations,
                        "learning_rate": lr,
                        "depth": depth,
                        "classes_count": len(all_classes),
                        "early_stopping_rounds": int(iterations * 0.8) + 1,
                        "use_best_model": True,
                        "random_seed": SEED,
                        "verbose": 0,
                        "loss_function": "MultiClass",
                    }

                    # Обучаем модель
                    current_model = self.train(
                        params=params,
                        y=y,
                        to_drop=to_drop,
                        cat_idxs=cat_idxs,
                        df_train=df_train,
                        df_test=df_test,
                        test_pool=test_pool,
                        all_data=all_data,
                    )

                    # Оцениваем accuracy на тестовой выборке
                    # test_pool = Pool(
                    #     X_test,
                    #     label=y_test,
                    #     cat_features=cat_idxs,
                    # )
                    preds = current_model.predict(test_pool, prediction_type="Class")
                    acc, f1 = eval_model(df_test[f"{y}"], preds)
                    if USE_DETAILED_LOG:
                        logger.info(f"Accuracy: {acc:.4f}")

                    # Сохраняем лучшую модель
                    if acc > best_score:
                        best_score = acc
                        best_model = current_model
                        best_params = params
                        if USE_DETAILED_LOG:
                            logger.info(
                                f"*** New best model! Accuracy: {acc:.4f} *** Best params: {best_params}"
                            )
        return best_model

    def train_on_field(
        self, df: pd.DataFrame, y: str, to_drop: list[str], parameters: dict
    ):
        """train model on a given field y"""
        if USE_DETAILED_LOG:
            logger.info(f"Predict {len(df[y])} records")

        bscores = []
        # total_preds = pd.DataFrame(index=df.index)
        df = df.copy()
        if y == "cash_flow_details_code":
            self.field_models[y] = {}
            for item in df["cash_flow_item_code"].unique():
                # model = None
                if USE_DETAILED_LOG:
                    logger.info(
                        f"Item {item}: found {len(df[df['cash_flow_item_code'] == item])} records"
                    )
                # group and get df_i sample
                df_i = df[df["cash_flow_item_code"] == item].copy()

                det_map1 = {name: num for num, name in enumerate(df_i[y].unique())}
                det_unmap1 = {v: k for k, v in det_map1.items()}

                df_i[f"{y}_norm"] = df_i[y].copy().map(det_map1)

                to_drop = [y for y in to_drop if y in df_i.columns]
                to_drop.extend(
                    [
                        c
                        for c in self.str_columns
                        if c in df_i.columns and c not in self.categorical
                    ]
                )

                df_i.drop(to_drop, inplace=True, axis=1)
                # df_i.fillna("empty", inplace=True)

                all_data = make_all_data(df_i, f"{y}")
                all_classes = all_data[f"{y}_norm"].unique()
                if USE_DETAILED_LOG:
                    logger.info("Total len of classes data: %s", len(all_data))

                if len(all_data) == 1:
                    this_label = det_unmap1.get(all_data[f"{y}_norm"].iloc[0])
                    bscores.extend([1 for _ in range(len(df_i[f"{y}_norm"]))])
                    # total_preds.extend([this_label for _ in range(len(df_i[f'{y}_norm']))])
                    # total_preds.loc[df_i.index, "model_pred"] = [
                    #     this_label for _ in range(len(df_i))
                    # ]
                    print("Only one details code, no need for model")
                    # TODO: сохранить словарь
                    self.strict_acc[y][item] = this_label
                    continue

                # rm txt from models inputs
                # to_drop = [
                #     "article_code",
                #     "base_document_date",
                #     "base_document_kind",
                #     "base_document_number",
                #     "base_document_operation_type",
                #     "article_document_date",
                #     "article_document_number",
                #     "article_name",
                # ]

                # X, y train
                df_test = df_i.sample(frac=0.05, random_state=SEED)
                df_train = df_i.drop(df_test.index)

                if len(df_test) == 0:
                    df_test = df_train.copy()

                df_test = self.make_full(
                    df_test, all_data, f"{y}_norm", 0, len(df_test)
                )
                df_train = self.make_full(
                    df_train, all_data, f"{y}_norm", 0, len(df_train)
                )
                # cats indxs
                cat_idxs = [
                    df_i.columns.get_loc(key=cat)
                    for cat in self.categorical
                    if cat in df_i.columns
                ]

                test_pool = self.get_test_pool(
                    df_test=df_test,
                    to_drop=[],
                    cat_idxs=cat_idxs,
                    all_data=all_data,
                    y=f"{y}_norm",
                )
                model_i = self.gridsearch(
                    all_classes=all_classes,
                    df_train=df_train,
                    df_test=df_test,
                    cat_idxs=cat_idxs,
                    test_pool=test_pool,
                    y=f"{y}_norm",
                    to_drop=[],
                    lr=parameters.get("lr", 0.01),
                    trees=parameters.get("trees", 30),
                    all_data=all_data,
                )
                self.field_models[y][item] = model_i
                self._save_cb_model(model_i, column=y, item=item)
        else:
            if USE_DETAILED_LOG:
                logger.info(f"Found {len(df[y])} records")
            det_map1 = {name: num for num, name in enumerate(df[y].unique())}
            det_unmap1 = {v: k for k, v in det_map1.items()}

            df[f"{y}_norm"] = df[y].copy().map(det_map1)

            to_drop = [y for y in to_drop if y in df.columns]
            to_drop.extend(
                [
                    c
                    for c in self.str_columns
                    if c in df.columns and c not in self.categorical
                ]
            )
            df.drop(to_drop, axis=1, inplace=True)

            all_data = make_all_data(df, f"{y}")
            all_classes = all_data[f"{y}_norm"].unique()
            if USE_DETAILED_LOG:
                logger.info("Total len of classes data: %s", len(all_data))

            if len(all_data) == 1:
                # TODO: get rid of this, save encoder to .pkl
                this_label = det_unmap1.get(all_data[f"{y}_norm"].iloc[0])
                bscores.extend([1 for _ in range(len(df[f"{y}_norm"]))])
                # total_preds.loc[df.index, f"{y}_pred"] = [
                #     this_label for _ in range(len(df))
                # ]
                # (f"Only one {y} code, no need for model")
            # X, y train
            df_test = df.sample(frac=0.05, random_state=SEED)
            df_train = df.drop(df_test.index)

            cat_idxs = [
                df.columns.get_loc(key=cat)
                for cat in self.categorical
                if cat in df.columns
            ]
            if len(df_test) == 0:
                df_test = df_train.copy()

            df_test = self.make_full(df_test, all_data, f"{y}_norm", 0, len(df_test))
            df_train = self.make_full(df_train, all_data, f"{y}_norm", 0, len(df_train))

            test_pool = self.get_test_pool(
                df_test=df_test,
                to_drop=[],
                y=f"{y}_norm",
                cat_idxs=[
                    df_test.columns.get_loc(key=cat)
                    for cat in self.categorical
                    if cat in df_test.columns
                ],
                all_data=all_data,
            )
            model_i = self.gridsearch(
                all_classes=all_classes,
                df_train=df_train,
                df_test=df_test,
                cat_idxs=cat_idxs,
                test_pool=test_pool,
                y=f"{y}_norm",
                to_drop=[],
                lr=parameters.get("lr", 0.01),
                trees=parameters.get("trees", 30),
                all_data=all_data,
            )
            self.field_models[y] = model_i
            self._save_cb_model(model_i, y)

    async def _fit(self, df: pd.DataFrame, parameters: dict, is_first=True):
        is_first = True  # TODO: ?
        if USE_DETAILED_LOG:
            logger.info("{} fit".format("First" if is_first else "continuous"))

        for y in self.y_columns:
            self.strict_acc[y] = {}
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y))
                logger.info(
                    f"Columns to predict: {self.y_columns}, got columns: {df.columns}, shape: {df.shape}"
                )
            self.train_on_field(
                df=df, y=y, to_drop=self.y_columns, parameters=parameters
            )
            gc.collect()
            # self._save_cb_model(model, column=y, item=item)
            # gc.collect()
            # self._load_all_models()

    async def fit(self, Xy_api: dict, parameters):
        X_y = pd.DataFrame(Xy_api)
        logger.info("Fitting")
        try:
            need_to_initialize = (
                self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR]
                or parameters.get("refit") == 0
            )
            calculate_metrics = parameters.get("calculate_metrics")
            use_cross_validation = parameters.get("use_cross_validation")

            await self._before_fit(
                parameters, need_to_initialize, calculate_metrics, use_cross_validation
            )

            train_test_indexes = None
            self.metrics_dataset_name = ""
            self.test_metrics_dataset_name = ""
            if use_cross_validation:
                train_test_indexes = self._get_train_test_indexes(X_y)
                if calculate_metrics:
                    self.metrics_dataset_name = await self._save_dataset_to_temp(
                        X_y.iloc[train_test_indexes[0]]
                    )
                    self.test_metrics_dataset_name = await self._save_dataset_to_temp(
                        X_y.iloc[train_test_indexes[1]]
                    )
            else:
                if calculate_metrics:
                    self.metrics_dataset_name = await self._save_dataset_to_temp(X_y)

            datasets = await self._transform_dataset(
                X_y,
                parameters,
                need_to_initialize,
                train_test_indexes,
                calculate_metrics,
            )

            await self._fit(datasets["train"], parameters, is_first=need_to_initialize)

            if calculate_metrics:
                await self._calculate_metrics(
                    parameters, need_to_initialize, use_cross_validation
                )

            await self._after_fit(
                parameters,
                need_to_initialize=need_to_initialize,
                use_cross_validation=use_cross_validation,
            )

        except Exception as e:
            await self._on_fitting_error(e)

    def __save(self, model: CatBoostClassifier, path: str):
        model.save_model(path)

    async def predict(self, X: pd.DataFrame, for_metrics=False):
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError("Model is not ready. Fit it before.")
        # load for details code
        field_models = self.field_models
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")
        X = pd.DataFrame(X)
        row_numbers = list(X.index)
        X_result = X.copy()

        pipeline_list = []
        pipeline_list.append(("checker", Checker(self.parameters, for_predict=True)))
        pipeline_list.append(
            ("nan_processor", NanProcessor(self.parameters, for_predict=True))
        )
        pipeline_list.append(
            ("feature_addder", FeatureAdder(self.parameters, for_predict=True))
        )
        if self.need_to_encode:
            pipeline_list.append(("data_encoder", self.data_encoder))

        pipeline = Pipeline(pipeline_list)

        for y in self.y_columns:
            X[y] = ""
        # set_config(transform_output="pandas")
        X_y = pipeline.fit_transform(X).copy()

        # c_x_columns = self.x_columns + [
        #     "number",
        #     "date",
        #     # "pred_cash_flow_item_name",
        #     # "pred_cash_flow_details_name",
        # ]

        for y in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start predicting. Field = "{}"'.format(y))
            if y == "cash_flow_details_code":
                cash_flow_items = list(X_y["cash_flow_item_code"].unique())
                # X_y["row_number"] = row_numbers
                X_y_list = []

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:
                        logger.info("Predicting {} - {}".format(ind, item_col))

                    Xy1 = X_y.loc[X_y["cash_flow_item_code"] == item_col].copy()

                    if (
                        self.strict_acc.get(y) is not None
                        and self.strict_acc[y].get(item_col) is not None
                    ):
                        # No model nedeed
                        Xy1[y] = self.strict_acc[y][item_col]
                    else:
                        model = field_models[y][str(item_col)]

                        X = Xy1[self.x_columns].to_numpy()
                        Xy1 = Xy1[[feat for feat in model.feature_names_]]
                        cat_idxs = [
                            Xy1.columns.get_loc(key=cat)
                            for cat in self.categorical
                            if cat in Xy1.columns
                        ]

                        X_pool = Pool(
                            Xy1,
                            cat_features=cat_idxs,
                        )

                        y_pred = model.predict(X_pool, prediction_type="Class")
                        # Xy1[y] = y_pred.ravel()

                        X_y[X_y["cash_flow_item_code"] == item_col][y] = y_pred.ravel()
            else:
                # c_y_columns = [y]
                # TODO: наверное, это если словарь
                if self.strict_acc.get(y) is not None:
                    X_y[y] = self.strict_acc[y]
                else:
                    X = X_y.copy()
                    model = field_models[y]
                    if "pred_cash_flow_item_name" in X.columns:
                        ymap = {
                            x: i
                            for i, x in enumerate(
                                X["pred_cash_flow_item_name"].unique()
                            )
                        }
                        # TODO: это в декодер
                        X[f"{y}_norm"] = X["pred_cash_flow_item_name"].map(ymap)
                    # to_drop = self.y_columns
                    else:
                        logging.error("No results for {y} from fasttext")
                    X = X[[feat for feat in model.feature_names_]]
                    cat_idxs = [
                        X.columns.get_loc(key=cat)
                        for cat in self.categorical
                        if cat in X.columns
                    ]

                    X_pool = Pool(
                        X,
                        cat_features=cat_idxs,
                    )
                    if USE_DETAILED_LOG:
                        logging.info(f"Feature names: {model.feature_names_}")

                    predictions = model.predict(X_pool, prediction_type="Class")
                    X_y[y] = predictions.ravel()
                if USE_DETAILED_LOG:
                    logger.info('Predicting model. Field = "{}". Done'.format(y))

            # c_x_columns = c_x_columns + c_y_columns

        # if self.need_to_encode:
        #     X_y = pipeline.named_steps["data_encoder"].inverse_transform(X_y)

        for y in self.y_columns:
            X_result[y] = X_y[y]

        if for_metrics:
            return X_result
        else:
            return X_result.to_dict(orient="records")

    def _get_cb_model(self, parameters):
        epochs = parameters.get("epochs", 20)
        depth = parameters.get("depth", 8)
        model = CatBoostClassifier(
            iterations=epochs,
            learning_rate=0.1,
            depth=depth,
            thread_count=THREAD_COUNT,
            used_ram_limit=USED_RAM_LIMIT,
        )

        return model

    def make_full(
        self, df: pd.DataFrame, all_data: pd.DataFrame, y: str, i=0, j=None
    ) -> pd.DataFrame:
        """
        fills dataset for each batch (df[i:j]) to have every class presented in all_classes
        """
        if j is None:
            j = len(df)

        classes_df = df[i:j].groupby(df[y]).first().copy()
        classes_df.reset_index(drop=True, inplace=True)

        if not all_data.loc[all_data.index.difference(classes_df.index)].empty:
            Xy = pd.concat(
                [
                    df[i:j],
                    all_data.loc[all_data.index.difference(classes_df[f"{y}"])],
                ],
                ignore_index=True,
            )
        else:
            Xy = df[i:j]
        return Xy

    def get_batch_pool(
        self,
        df: pd.DataFrame,
        y: str,
        all_data: pd.DataFrame,
        to_drop: list[str],
        cat_idxs: list[int],
        batch_size: int = 1000,
    ):
        """Batches Pool generator"""
        i = 0
        j = 0
        while i < len(df):
            j = min(i + batch_size, len(df))
            Xy = self.make_full(df=df, all_data=all_data, y=f"{y}", i=i, j=j)
            y_batch = Xy[f"{y}"]
            X_batch = Xy.drop([y for y in self.y_columns if y in Xy.columns], axis=1)
            X_batch = Xy.drop(to_drop, axis=1)
            pool = Pool(
                X_batch,
                label=y_batch,
                cat_features=cat_idxs,
            )
            # if QUANTIZE:
            #     pool.quantize()
            yield pool
            del Xy, y_batch, X_batch
            gc.collect()
            i = j

    def get_test_pool(
        self,
        df_test: pd.DataFrame,
        y: str,
        all_data,
        to_drop: list[str],
        cat_idxs: list[int],
    ):
        # self.field_models[y]
        self.make_full(df=df_test, y=y, all_data=all_data, i=0, j=len(df_test))
        y_test = df_test[f"{y}"]
        # X_test = df_test.drop([y for y in self.y_columns if y in df_test.columns], axis=1)
        X_test = df_test.drop(to_drop, axis=1)
        X_test.columns
        # text_features=txts)
        pool = Pool(
            X_test,
            label=y_test,
            cat_features=cat_idxs,
        )
        # if QUANTIZE:
        #     pool.quantize()

        return pool

    def _load_cb_model(self, column, item=None, number=None) -> CatBoostClassifier:
        path_to_model_folder = os.path.join(MODEL_FOLDER, self.uid, column)
        if item is not None:
            path_to_model = os.path.join(path_to_model_folder, "{}.cbm".format(item))
        elif number is not None:
            path_to_model = os.path.join(
                path_to_model_folder, "{}.cbm".format(str(number))
            )
        else:
            path_to_model = os.path.join(path_to_model_folder, "sum.cbm")

        if not os.path.isdir(path_to_model_folder):
            os.makedirs(path_to_model_folder)
        model = CatBoostClassifier()
        model.load_model(path_to_model)
        if item is not None:
            if not self.field_models.get(column):
                self.field_models[column] = {}
            self.field_models[column][str(item)] = model
        else:
            self.field_models[column] = model

    def _load_column_models(self, column, sum_model=False):
        folder = os.path.join(MODEL_FOLDER, self.uid, column)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filenames = os.listdir(folder)
        models = {}

        for filename in filenames:
            if filename == "sum.cbm" and not sum_model:
                continue
            if filename != "sum.cbm" and sum_model:
                continue

            item = filename.split(".")[0]
            model = CatBoostClassifier()
            model.load_model(os.path.join(MODEL_FOLDER, self.uid, column, filename))
            models[item] = model

        return models

    def _load_all_models(self):
        for y in self.y_columns:
            models = self._load_column_models(
                y, sum_model=y != "cash_flow_details_code"
            )
            if models:
                if y != "cash_flow_details_code":
                    self.field_models[y] = list(models.values())[0]
                else:
                    for item, model in models.items():
                        self.field_models[y][item] = model

    def _delete_submodels(self, column):
        path_to_dir = os.path.join(MODEL_FOLDER, self.uid, column)

        filenames = os.listdir(path_to_dir)
        for filename in filenames:
            if filename == "sum.cbm":
                continue

            os.remove(os.path.join(path_to_dir, filename))

    async def _transform_dataset(
        self,
        dataset,
        parameters,
        need_to_initialize,
        train_test_indexes=None,
        calculate_metrics=False,
    ):
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")

        pipeline_list = []
        pipeline_list.append(("checker", Checker(self.parameters)))
        pipeline_list.append(("nan_processor", NanProcessor(self.parameters)))
        pipeline_list.append(("feature_adder", FeatureAdder(self.parameters)))
        if self.need_to_encode:
            if need_to_initialize:
                self.data_encoder = DataEncoder(self.parameters)
            self.data_encoder.form_encode_dict = need_to_initialize
            pipeline_list.append(("data_encoder", self.data_encoder))

        # pipeline_list.append(("shuffler", Shuffler(self.parameters)))

        pipeline = Pipeline(pipeline_list)
        dataset = pipeline.fit_transform(dataset)

        datasets = {}

        if train_test_indexes:
            datasets["train"] = dataset.iloc[train_test_indexes[0]]
            datasets["test"] = dataset.iloc[train_test_indexes[1]]
        else:
            datasets["train"] = dataset

        self.classes = {}
        for y in self.y_columns:
            self.classes[y] = list(datasets["train"][y].unique())
        gc.collect()
        return datasets

    async def _on_fitting_error(self, ex):
        self._delete_all_models()
        await super()._on_fitting_error(ex)
