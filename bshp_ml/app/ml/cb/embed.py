from .classifier import CatBoostModel, CbCallBack

from abc import ABC, abstractmethod
import logging
from typing import Optional
from enum import Enum
import pickle
from datetime import datetime, UTC
import os
import uuid
import pandas as pd
import numpy as np

import gc
from .models import Model
import shutil
from copy import deepcopy
import json
import random

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from bshp_ml.app.ml.cb.classifier import (
    CatBoostClassifier,
    Pool,
    sum_models,
    to_classifier,
)

from data_processing import (
    Reader,
    Checker,
    DataEncoder,
    NanProcessor,
    Shuffler,
    FeatureAdder,
    data_loader,
)
from db import db_processor
from schemas.tasks import ModelStatuses, ModelTypes
from settings import (
    MODEL_FOLDER,
    THREAD_COUNT,
    USE_DETAILED_LOG,
    USED_RAM_LIMIT,
    DATASET_BATCH_LENGTH,
    QUANTIZE,
)

logging.getLogger("bshp_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def train(params: dict, df_train, df_test):
    model = None
    model_new = None
    metrics = []
    metrics_test = []
    test_pool = get_test_pool(df_test)

    for n, batch in enumerate(get_batch_pool(df=df_train, batch_size=25_000)):
        print(f"BATCH {n} of total {len(df_train) // 25_000 + 1}")
        if model is None:
            model = CatBoostClassifier(**params)
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
            )

        acc, f1 = eval_model(
            batch.get_label(), model.predict(batch, prediction_type="Class")
        )
        metrics.append((acc, f1))
        acc, f1 = eval_model(y_test, model.predict(test_pool, prediction_type="Class"))
        metrics_test.append((acc, f1))
        free()

    show_metrics(metrics, metrics_test)
    show_importance(model)
    return model


param_grid = {
    "learning_rate": [0.002, 0.004],
    "depth": [5],
    "iterations": [300, 400],
    "l2_leaf_reg": [3],
}

best_score = 0
best_model = None
best_params = None

# Перебираем все комбинации параметров
for lr in param_grid["learning_rate"]:
    for depth in param_grid["depth"]:
        for iterations in param_grid["iterations"]:
            print(f"\n=== Testing: lr={lr}, depth={depth}, iterations={iterations} ===")

            params = {
                "task_type": "GPU",
                "iterations": iterations,
                "learning_rate": lr,
                "depth": depth,
                "classes_count": len(all_classes),
                "early_stopping_rounds": 90,
                "use_best_model": True,
                "random_seed": SEED,
                "verbose": 0,
            }

            # Обучаем модель
            current_model = train(params, df_train, df_test)

            # Оцениваем accuracy на тестовой выборке
            test_pool = Pool(
                X_test,
                label=y_test,
                cat_features=cats,
            )
            preds = current_model.predict(test_pool, prediction_type="Class")
            acc, f1 = eval_model(y_test, preds)

            print(f"Accuracy: {acc:.4f}")

            # Сохраняем лучшую модель
            if acc > best_score:
                best_score = acc
                best_model = current_model
                best_params = params
                print(f"*** New best model! Accuracy: {acc:.4f} ***")

print(f"\n=== BEST PARAMS ===")
print(f"Learning rate: {best_params['learning_rate']}")
print(f"Depth: {best_params['depth']}")
print(f"Iterations: {best_params['iterations']}")
print(f"Best accuracy: {best_score:.4f}")

# Сохраняем лучшую модель
name = f"{col}_embeddings_best"
best_model.save_model(f"/kaggle/working/{name}.cbm")
print(f"Best model saved as: {name}.cbm")


class CatBoostModelEmbeddings(CatBoostModel):
    """Класс модели catboost с пониманием эмбеддингов"""

    async def _fit(self, dataset, parameters, is_first=True):
        if USE_DETAILED_LOG:
            logger.info("{} fit".format("First" if is_first else "continuous"))

        pools = dataset

        if is_first:
            self.strict_acc = {}
            self.test_strict_acc = {}

        c_x_columns = self.x_columns.copy()

        indexes_to_encode = []
        for ind, col in enumerate(self.x_columns):
            if col in self.columns_to_encode:
                indexes_to_encode.append(ind)

        for y_col in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))
            if y_col != "cash_flow_details_code":
                pool = pools[y_col]
                if isinstance(pool, list):
                    models = []

                    for ind, c_pool in enumerate(pool):
                        if USE_DETAILED_LOG:
                            logger.info("Model {}".format(ind))

                        c_model = self._get_cb_model(parameters)
                        c_model.set_params(class_names=self.classes[y_col])
                        c_model.fit(c_pool, callbacks=[CbCallBack()], verbose=False)
                        self._save_cb_model(c_model, y_col, number=ind)
                        del c_model
                        gc.collect()

                    models = list(self._load_column_models(y_col).values())
                    if not is_first:
                        models = [self.field_models[y_col]] + models
                    model = (
                        to_classifier(sum_models(models))
                        if len(models) > 1
                        else models[0]
                    )
                    self.field_models[y_col] = model
                    self._save_cb_model(model, y_col)
                    del model
                    for c_model in models:
                        del c_model
                    gc.collect()

                    self._delete_submodels(y_col)

                elif isinstance(pool, Pool):
                    init_model = self.field_models[y_col] if not is_first else None
                    c_model = self._get_cb_model(parameters)
                    c_model.fit(
                        pool,
                        callbacks=[CbCallBack()],
                        verbose=False,
                        init_model=init_model,
                    )
                    self._save_cb_model(c_model, y_col)
                    del c_model
                    gc.collect()
                else:
                    self.strict_acc[y_col] = pool
            else:
                self.field_models[y_col] = {}
                self.strict_acc[y_col] = {}

                c_pools = pools[y_col]

                ind = 0
                for item_col, c_pool in c_pools.items():
                    if USE_DETAILED_LOG:
                        logger.info("Fitting {} - {}".format(ind, item_col))
                    if isinstance(c_pool, Pool):
                        init_model = (
                            self.field_models[y_col][item_col] if not is_first else None
                        )
                        c_model = self._get_cb_model(parameters)

                        c_model.fit(
                            c_pool,
                            callbacks=[CbCallBack()],
                            verbose=False,
                            init_model=init_model,
                        )

                        self._save_cb_model(c_model, column=y_col, item=item_col)
                        del c_model
                        gc.collect()
                    else:
                        self.strict_acc[y_col][item_col] = c_pool

                    ind += 1
            c_x_columns.append(y_col)
            indexes_to_encode.append(len(c_x_columns))

            self._load_all_models()

    async def predict(self, X, for_metrics=False):
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError("Model is not ready. Fit it before.")

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

        for y_col in self.y_columns:
            X[y_col] = ""

        X_y = pipeline.transform(X).copy()

        c_x_columns = self.x_columns

        for y_col in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start predicting. Field = "{}"'.format(y_col))
            if y_col == "cash_flow_details_code":
                cash_flow_items = list(X_y["cash_flow_item_code"].unique())
                X_y["row_number"] = row_numbers
                X_y_list = []

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:
                        logger.info("Predicting {} - {}".format(ind, item_col))

                    c_X_y = X_y.loc[X_y["cash_flow_item_code"] == item_col].copy()

                    if (
                        self.strict_acc.get(y_col) is not None
                        and self.strict_acc[y_col].get(item_col) is not None
                    ):
                        c_X_y[y_col] = self.strict_acc[y_col][item_col]
                    else:
                        X = c_X_y[self.x_columns].to_numpy()

                        c_model = field_models[y_col][str(item_col)]
                        y_pred = c_model.predict(X)
                        c_X_y[y_col] = y_pred.ravel()

                    X_y_list.append(c_X_y)

                t_X_y = pd.concat(X_y_list, axis=0)
                if y_col in X_y.columns:
                    X_y = X_y.drop([y_col], axis=1)

                X_y = X_y.merge(
                    t_X_y[["row_number", y_col]], on=["row_number"], how="left"
                )
                X_y = X_y.set_index(X_y["row_number"])
            else:
                c_y_columns = [y_col]
                if self.strict_acc.get(y_col) is not None:
                    X_y[y_col] = self.strict_acc[y_col][item_col]
                else:
                    X = X_y[c_x_columns].to_numpy()

                    model = field_models[y_col]
                    y = model.predict(X)
                    X_y[y_col] = y.ravel()
                if USE_DETAILED_LOG:
                    logger.info('Predicting model. Field = "{}". Done'.format(y_col))

            c_x_columns = c_x_columns + c_y_columns

        if self.need_to_encode:
            X_y = pipeline.named_steps["data_encoder"].inverse_transform(X_y)

        for col in self.y_columns:
            X_result[col] = X_y[col]

        if for_metrics:
            return X_result
        else:
            return X_result.to_dict(orient="records")

    def _get_dataset_with_right_classes(
        self, dataset, x_columns, y_column, model_classes=None, all_dataset=None
    ):
        data_classes = set(dataset[y_column].unique())

        if model_classes is None and len(data_classes) <= 1:
            return None
        elif model_classes is not None:
            model_classes = set(model_classes)

            if data_classes == model_classes:
                result = dataset[x_columns + [y_column]]
            else:
                to_delete = []
                to_add = []
                for cl in model_classes:
                    if cl not in data_classes:
                        to_add.append(cl)
                for cl in data_classes:
                    if cl not in model_classes:
                        to_delete.append(cl)

                if to_delete:
                    result = dataset.loc[~dataset[y_column].isin(to_delete)]
                else:
                    result = dataset

                if to_add:
                    to_concat = []
                    for val in to_add:
                        add_dataset = all_dataset.loc[all_dataset[y_column] == val]
                        if len(add_dataset) > 0:
                            if len(add_dataset) >= 50:
                                add_dataset = add_dataset.iloc[:50]
                            to_concat.append(add_dataset)
                        else:
                            none_str = data_loader.get_none_data_row(self.parameters)
                            none_str[y_column] = val
                            to_concat.append(none_str)

                    result = pd.concat([result] + to_concat)

                print(to_delete, to_add)

                result = result[x_columns + [y_column]]
        else:
            result = dataset[x_columns + [y_column]]

        return result

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

    def _get_data_pools(self, dataset, test_dataset=None):
        pools = {}
        test_pools = None
        use_cross_validation = test_dataset is not None
        c_x_columns = self.x_columns.copy()
        to_delete = []
        cash_flow_items = list(dataset["cash_flow_item_code"].unique())
        for y_col in self.y_columns:
            if y_col != "cash_flow_details_code":
                value_items = list(dataset[y_col].unique())

                if DATASET_BATCH_LENGTH > 0 and len(value_items) > 1:
                    begin = 0
                    end = DATASET_BATCH_LENGTH
                    pools[y_col] = []
                    while True:
                        b_dataset = dataset.iloc[begin : min([end, len(dataset)])]
                        c_dataset = self._get_dataset_with_right_classes(
                            b_dataset,
                            c_x_columns,
                            y_col,
                            model_classes=value_items,
                            all_dataset=dataset,
                        )
                        c_pool = Pool(c_dataset[c_x_columns], c_dataset[y_col])
                        if QUANTIZE:
                            c_pool.quantize()
                        pools[y_col].append(c_pool)
                        to_delete.append(c_dataset)
                        to_delete.append(b_dataset)
                        if end >= len(dataset):
                            break
                        begin += DATASET_BATCH_LENGTH
                        end += DATASET_BATCH_LENGTH
                else:
                    c_dataset = self._get_dataset_with_right_classes(
                        dataset, c_x_columns, y_col
                    )
                    if c_dataset is not None:
                        pools[y_col] = Pool(c_dataset[c_x_columns], c_dataset[y_col])
                        if QUANTIZE:
                            pools[y_col].quantize()
                        to_delete.append(c_dataset)
                    else:
                        pools[y_col] = dataset.iloc[0][y_col]

            else:
                pools[y_col] = {}
                for ind, item_col in enumerate(cash_flow_items):
                    c_dataset = dataset.loc[dataset["cash_flow_item_code"] == item_col]
                    value = c_dataset.iloc[0][y_col]
                    c_dataset = self._get_dataset_with_right_classes(
                        c_dataset, c_x_columns, y_col
                    )
                    if c_dataset is not None:
                        pools[y_col][item_col] = Pool(
                            c_dataset[c_x_columns], c_dataset[y_col]
                        )
                        pools[y_col][item_col].quantize()
                        to_delete.append(c_dataset)
                    else:
                        pools[y_col][item_col] = value

            c_x_columns.append(y_col)

        to_delete.append(dataset)

        for el in to_delete:
            del el
            el = pd.DataFrame()

        gc.collect()

        return pools, test_pools

    def _save_cb_model(self, model: CatBoostClassifier, column, item=None, number=None):
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

        model.save_model(path_to_model)

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
        for y_col in self.y_columns:
            models = self._load_column_models(
                y_col, sum_model=y_col != "cash_flow_details_code"
            )
            if models:
                if y_col != "cash_flow_details_code":
                    self.field_models[y_col] = list(models.values())[0]
                else:
                    for item, model in models.items():
                        self.field_models[y_col][item] = model

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
        datasets = await super()._transform_dataset(
            dataset,
            parameters,
            need_to_initialize,
            train_test_indexes,
            calculate_metrics,
        )
        pools = {}
        self.classes = {}
        if need_to_initialize:
            for y_col in self.y_columns:
                self.classes[y_col] = list(datasets["train"][y_col].unique())
        train_pools, test_pools = self._get_data_pools(
            datasets["train"], datasets.get("test")
        )
        del datasets["train"]
        datasets["train"] = pd.DataFrame()
        pools["train"] = train_pools

        if datasets.get("test") is not None:
            del datasets["test"]
            datasets["test"] = pd.DataFrame()

        gc.collect()

        return pools

    async def _on_fitting_error(self, ex):
        self._delete_all_models()
        await super()._on_fitting_error(ex)

    def _save_column_model(self, column, item=None):
        if not os.path.isdir(os.path.join(MODEL_FOLDER, self.uid, column)):
            os.makedirs(os.path.join(MODEL_FOLDER, self.uid, column))

        if item is not None:
            if self.strict_acc.get(column) and self.strict_acc[column].get(item):
                value = self.strict_acc[column][item]
                with open(
                    os.path.join(
                        MODEL_FOLDER, self.uid, column, "{}.json".format(item)
                    ),
                    "w",
                ) as fp:
                    json.dump({"value": int(value)}, fp)
            elif self.field_models.get(column) and self.field_models.get(column).get(
                str(item)
            ):
                self._save_cb_model(self.field_models[column][str(item)], column, item)
        else:
            if self.strict_acc.get(column):
                with open(
                    os.path.join(MODEL_FOLDER, self.uid, column, "sum.json"), "w"
                ) as fp:
                    value = self.strict_acc[column]
                    json.dump({"value": int(value)}, fp)
            elif self.field_models.get(column):
                self._save_cb_model(self.field_models[column], column)

    def _load_column_model(self, column, item=None):
        if item is not None:
            if os.path.exists(
                os.path.join(MODEL_FOLDER, self.uid, column, "{}.json".format(item))
            ):
                if not self.strict_acc.get(column):
                    self.strict_acc[column] = {}
                with open(
                    os.path.join(MODEL_FOLDER, self.uid, column, "{}.json".format(item))
                ) as fp:
                    self.strict_acc[column][item] = np.int64(json.load(fp)["value"])
            elif os.path.exists(
                os.path.join(MODEL_FOLDER, self.uid, column, "{}.cbm".format(item))
            ):
                self._load_cb_model(column, item)
        else:
            if os.path.exists(os.path.join(MODEL_FOLDER, self.uid, column, "sum.json")):
                with open(
                    os.path.join(MODEL_FOLDER, self.uid, column, "sum.json")
                ) as fp:
                    self.strict_acc[column] = np.int64(json.load(fp)["value"])
            elif os.path.exists(
                os.path.join(MODEL_FOLDER, self.uid, column, "sum.cbm")
            ):
                self._load_cb_model(column)
