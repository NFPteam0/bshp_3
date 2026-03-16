import asyncio
import copy
from functools import lru_cache
import gc
import json
import logging
import os

from fastapi import HTTPException
from sklearn.pipeline import Pipeline
from .utils import prepare_sentences, preprocess_text
import numpy as np
from gensim.models import FastText
from ..models import Model, get_model_manager
from ml.data_processing import (
    Checker,
    DataEncoder,
    FeatureAdder,
    NanProcessor,
    Shuffler,
)
from schemas.models import EmbedPredictionsRow, ModelStatuses, ModelTypes
import pandas as pd
from settings import (
    MODEL_FOLDER,
    THREAD_COUNT,
    USE_DETAILED_LOG,
    USED_RAM_LIMIT,
    DATASET_BATCH_LENGTH,
    QUANTIZE,
)

# prep_sents = [preprocess_text(" ".join(sent)).split() for sent in prepare_sentences(df, txt_cols)]

logging.getLogger("bshp_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

BATCH_SIZE = 20_000


class FastTextModel(Model):
    model_type = ModelTypes.fstxt
    _model: FastText | None = None
    status: ModelStatuses = ModelStatuses.CREATED

    all_classes_names: dict[str, list[str]] | None = None

    def __init__(self, base_name: str):
        super().__init__(base_name)
        self.str_columns.extend(
            [
                "cash_flow_item_name",
                "cash_flow_details_name",
                "payment_purpose",
                "kind",
                "payment_purpose_returned",
                "contract_name",
                "contract_number",
            ]
        )
        self.y_columns = [
            "cash_flow_item_name",
            "cash_flow_details_name",
            "year",
        ]
        self.need_to_encode = False

    async def load(self, uid):
        self.uid = uid

        if self.status != ModelStatuses.ERROR:
            # TODO: по опции from_pretrained=True?
            # ИНАЧЕ self._create, но нужно
            # предобработать, это не здесь

            try:
                df = await self._read_dataset(
                    parameters={
                        "data_filter": {}
                        # if self.base_name != "all_bases"
                        # else None
                    }
                )
                self.all_classes_names = {
                    col: df[col].unique() for col in self.y_columns
                }
                self.name2code = {
                    "cash_flow_item_name": "cash_flow_item_code",
                    "cash_flow_details_name": "cash_flow_details_code",
                    "year": "year",
                }
                self.all_classes_codes = {
                    col: dict(
                        zip(
                            df[col].unique(),
                            df[self.name2code[col]].replace("", -1).astype(int),
                        )
                    )
                    for col in self.y_columns
                }
                logger.info(
                    "Classes found: %s",
                    str({cls: len(lst) for cls, lst in self.all_classes_names.items()}),
                )
            except KeyError as e:
                # TODO: другой эксцепт
                logger.warning("No classes for embeddings detected: {e}")

            self._load_pretrained()

    def _load_pretrained(self, model_folder=MODEL_FOLDER):
        # NOTE: у fasttext есть ивенты из коробки
        self._model = FastText.load(f"{model_folder}/pretrained/fsttxt.model")

    def _sync_fit(self, df: pd.DataFrame, parameters, is_first=False):
        if USE_DETAILED_LOG:
            logger.info("{} fit".format("First" if is_first else "continuous"))
        is_first = False  # TODO: ?
        # if is_first:
        #     self.strict_acc = {}
        #     self.test_strict_acc = {}
        # c_x_columns = self.x_columns.copy()

        # indexes_to_encode = []
        # for ind, col in enumerate(self.x_columns):
        #     if col in self.columns_to_encode:
        #         indexes_to_encode.append(ind)
        X = df[self.str_columns].copy()
        logger.info("Shape of loaded data: %s", X.shape)
        # always False until TODO in .load
        if self._model is None:
            logger.info("No pretrained model found")
            # Если нет загруженной, нажали на fit
            # self._load_pretrained()
            # _create?
            ...
        if is_first or self.all_classes_names is None:
            self.all_classes_names = {col: df[col].unique() for col in self.y_columns}

            if USE_DETAILED_LOG:
                logger.info(
                    "Classes found: %s",
                    str({cls: len(lst) for cls, lst in self.all_classes_names.items()}),
                )

        if not is_first:
            # new_sentences = [
            #     preprocess_text(" ".join(sent)).split()
            #     for sent in prepare_sentences(X, self.str_columns)
            # ]
            new_sentences = prepare_sentences(X, self.str_columns)

            logger.info(
                "Training pretrained model with %d ngrams", self._model.word_ngrams
            )

            self._model.build_vocab(new_sentences, update=True)
            self._model.train(
                new_sentences,
                total_examples=len(new_sentences),
                epochs=self._model.epochs,
            )

    async def _fit(self, df: pd.DataFrame, parameters, is_first=False):
        return await asyncio.to_thread(self._sync_fit, df, parameters, is_first)

    def _get_all_classes(self, df: pd.DataFrame):
        # TODO: move to fit? no bare prefitted?
        ...

    def _create(self):
        # TODO: is_first = False
        """fits from scratch"""
        self._model = FastText(
            sentences=...,
            vector_size=2**7,
            window=30,
            min_count=1,
            workers=4,
            sg=1,
            min_n=2,
            max_n=8,
            epochs=15,
        )

    async def predict(
        self,
        X_api: dict | list[dict],
        for_metrics=False,
        set_classes: bool = False,
        set_from: dict = None,
    ) -> dict[str, EmbedPredictionsRow]:
        return await asyncio.to_thread(
            self._sync_predict, X_api, for_metrics, set_classes, set_from
        )

    def _sync_predict(
        self,
        X_api: dict | list[dict],
        for_metrics=False,
        set_classes: bool = False,
        set_from: dict = None,
    ) -> dict[str, EmbedPredictionsRow]:
        X = pd.DataFrame(X_api)
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")

        for col in ["cash_flow_item_name", "cash_flow_details_name"]:
            if col not in X.columns:
                X[col] = ""

        pipeline_list = []
        pipeline_list.append(("checker", Checker(self.parameters)))
        pipeline_list.append(("nan_processor", NanProcessor(self.parameters)))

        pipeline = Pipeline(pipeline_list)
        if USE_DETAILED_LOG:
            logger.warning("Predicting, Shape %s", X.shape)
        try:
            X = pipeline.fit_transform(X)
        except ValueError as e:
            logger.error("Can't transform empty data: X(%s)", X.shape)
            raise e
        # predict_detail
        # self.status != ModelStatuses.READY?
        if set_from is not None:
            try:
                set_from = pipeline.fit_transform(set_from)
            except ValueError as e:
                logger.error("Can't transform empty data: X(%s)", set_from.shape)
                raise e
        else:
            set_from = X
        if set_classes:
            self.all_classes_names = {
                col: set_from[col].unique() for col in self.y_columns
            }
            # TODO: коды сюда?
            if USE_DETAILED_LOG:
                logger.info(
                    "Classes found: %s",
                    str({cls: len(lst) for cls, lst in self.all_classes_names.items()}),
                )
            self.name2code = {
                "cash_flow_item_name": "cash_flow_item_code",
                "cash_flow_details_name": "cash_flow_details_code",
                "year": "year",
            }
            self.all_classes_codes = {
                col: dict(
                    zip(
                        set_from[col].unique(),
                        set_from[self.name2code[col]]
                        .replace("", -1)
                        .fillna(-1)
                        .astype(int),
                    )
                )
                for col in self.y_columns
            }
            if USE_DETAILED_LOG:
                for col in self.y_columns:
                    empty = set_from[self.name2code[col]].isna() | (
                        set_from[self.name2code[col]].astype(str).str.strip() == ""
                    )
                    if empty.any():
                        logger.warning(
                            f"Empty {self.name2code[col]} at number={set_from.loc[empty.idxmax(), 'number']}"
                        )
        if self.all_classes_names is None or self.all_classes_codes is None:
            raise ValueError(f"Model is not ready, it's {self.status}. Fit it before.")

        # X[self.y_columns] = ""

        # X = X_y[self.x_columns].to_numpy()
        # y = self._model.predict(X)
        # X_y[y_col] = y.ravel()
        # details cols
        UNFEATURED = [
            "company_inn",
            "company_kpp",
            "contractor_inn",
            "contractor_kpp",
            "company_account_number",
            "contractor_account_number",
            "cash_flow_item_code",
            "cash_flow_item_name",
            "cash_flow_details_code",
            "cash_flow_details_name",
            "year",
        ]

        sentences = prepare_sentences(
            X,
            [
                col
                for col in self.str_columns
                if col not in UNFEATURED and col not in self.y_columns
            ],
        )

        for y in self.y_columns:
            class_matrix, class_names = self.build_class_matrix(
                self.all_classes_names[y], self.all_classes_codes[y]
            )

            preds, probs = self.batched_predict(
                sentences, class_matrix, class_names, BATCH_SIZE
            )
            X[f"pred_{y}"] = preds
            X[f"prob_{y}"] = probs
            gc.collect()

        return X.to_json(
            orient="records", force_ascii=False, date_format="iso", date_unit="s"
        )

    def sentence_vector(self, words: list[str], wv):
        v = [wv[word] for word in words]
        return np.sum(v, axis=0)

    @staticmethod
    @lru_cache(maxsize=20_000)
    def wv_cached(word: str, base_name: str, model_type: str):
        model_manager = get_model_manager()
        model = model_manager.get_model(model_type, base_name, log=False)
        return model._model.wv[word]

    @staticmethod
    @lru_cache(maxsize=1000)
    def sentence_vector_cached(words: tuple[str], base_name: str, model_type: str):
        v = [FastTextModel.wv_cached(word, base_name, model_type) for word in words]
        return np.mean(v, axis=0)

    def build_class_matrix(self, classes, codes):
        vectors = []
        names = []
        for cls in classes:
            tokens = tuple((cls.lower()).split())
            vec = self.sentence_vector_cached(tokens, self.base_name, self.model_type)
            vectors.append(vec)
            names.append(cls)

        M = np.vstack(vectors).astype(np.float32)

        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9

        return M, names

    def batched_predict(
        self,
        sentences,
        class_matrix,
        class_names,
        chunk_size=2**12,
    ):
        n = len(sentences)

        pred_labels = []
        pred_probs = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            chunk = sentences[start:end]

            X = np.vstack(
                [
                    self.sentence_vector_cached(
                        tuple(s), self.base_name, self.model_type
                    )
                    for s in chunk
                ]
            ).astype(np.float32)

            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

            sims = X @ class_matrix.T

            idx = sims.argmax(axis=1)
            probs = sims.max(axis=1)

            pred_labels.extend(class_names[i] for i in idx)
            pred_probs.extend(probs.tolist())

            del X, sims
            gc.collect()
            if USE_DETAILED_LOG:
                logging.info(f"Fasttext model BATCH {start // chunk_size + 1} ready ")

        return pred_labels, pred_probs

    def _save_column_model(self, column, item=None):
        pass

    def _load_column_model(self, column, item=None):
        pass

    async def _transform_dataset(
        self,
        dataset: pd.DataFrame,
        parameters,
        need_to_initialize,
        train_test_indexes=None,
        calculate_metrics=False,
    ):
        # TODO: отдельно check_data? Но это же в другом слое..?
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")

        for col in ["cash_flow_item_name", "cash_flow_details_name"]:
            if col not in dataset.columns:
                dataset[col] = ""

        pipeline_list = []
        pipeline_list.append(("checker", Checker(self.parameters)))
        pipeline_list.append(("nan_processor", NanProcessor(self.parameters)))
        # pipeline_list.append(("feature_adder", FeatureAdder(self.parameters)))
        # if self.need_to_encode:
        #     if need_to_initialize:
        #         self.data_encoder = DataEncoder(self.parameters)
        #     self.data_encoder.form_encode_dict = need_to_initialize
        #     pipeline_list.append(("data_encoder", self.data_encoder))

        # pipeline_list.append(("shuffler", Shuffler(self.parameters)))

        pipeline = Pipeline(pipeline_list)
        dataset = pipeline.fit_transform(dataset)

        datasets = {}

        if train_test_indexes:
            datasets["train"] = dataset.iloc[train_test_indexes[0]]
            datasets["test"] = dataset.iloc[train_test_indexes[1]]
        else:
            datasets["train"] = dataset

        return datasets

    async def save(self, without_models=False):
        logging.info("Save model in %s", MODEL_FOLDER)
