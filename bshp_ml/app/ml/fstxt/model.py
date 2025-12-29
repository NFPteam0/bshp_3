import copy
from functools import lru_cache
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


class FastTextModel(Model):
    model_type = ModelTypes.fstxt
    _model: FastText | None = None
    status: ModelStatuses = ModelStatuses.CREATED

    all_classes_names: dict[str, list[str]] | None = None

    def __init__(self, base_name: str):
        super().__init__(base_name)
        self.str_columns.extend(
            ["cash_flow_item_name", "cash_flow_details_name", "payment_purpose", "kind"]
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
                logger.info(
                    "Classes found: %s",
                    str({cls: len(lst) for cls, lst in self.all_classes_names.items()}),
                )
            except Exception as e:
                # TODO: другой эксцепт
                logger.error("No classes for embeddings detected")
                raise ValueError(f"No classes for embeddings detected due to: {e}")

            self._load_pretrained()

    def _load_pretrained(self, model_folder=MODEL_FOLDER):
        # NOTE: у fasttext есть ивенты из коробки
        self._model = FastText.load(f"{model_folder}/pretrained/fsttxt.model")

    async def _fit(self, df: pd.DataFrame, parameters, is_first=False):
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
        self, X_api: dict | list[dict], for_metrics=False, set_classes: bool = False
    ) -> dict[str, EmbedPredictionsRow]:
        X = pd.DataFrame(X_api)
        # predict_detail
        # self.status != ModelStatuses.READY?
        if set_classes:
            self.all_classes_names = self.all_classes_names = {
                col: X[col].unique() for col in self.y_columns
            }
            if USE_DETAILED_LOG:
                logger.info(
                    "Classes found: %s",
                    str({cls: len(lst) for cls, lst in self.all_classes_names.items()}),
                )
        if self.all_classes_names is None:
            raise ValueError(f"Model is not ready, it's {self.status}. Fit it before.")

        # X[self.y_columns] = ""

        # X = X_y[self.x_columns].to_numpy()
        # y = self._model.predict(X)
        # X_y[y_col] = y.ravel()
        # details cols
        tmp_cols = []
        all_sentences = prepare_sentences(X, self.str_columns + tmp_cols)
        sentences = all_sentences
        result = {}
        for y in self.y_columns:
            res = []
            sentences_i = None
            if y == "cash_flow_details_name":
                # result.get("cash_flow_item_name")
                tmp_cols.append("pred_cash_flow_item_name")

                if "pred_cash_flow_item_name" in X.columns:
                    sentences_i = prepare_sentences(X, tmp_cols)
                else:
                    logger.warning("No predictions for items")
                if sentences_i:
                    sentences = [
                        sent + sent_i for sent, sent_i in zip(sentences, sentences_i)
                    ]
            else:
                sentences = all_sentences
            all_classes_names = self.all_classes_names[y]
            if USE_DETAILED_LOG:
                logging.info("Predicting %s", y)
                logging.info("Overall classes %d", len(self.all_classes_names[y]))
                logging.info("Feed model with %d txt columns", len(X.columns))
            wordvec = {cls: self._model.wv[cls] for cls in all_classes_names}
            vectors = np.vstack(list(wordvec.values()))
            words = list(wordvec.keys())
            # TODO: add validation?
            for sentence in sentences:
                # target_vector = self.sentence_vector(sentence, self._model.wv)
                target_vector = self.sentence_vector_cached(
                    tuple(sentence), self.base_name, self.model_type
                )

                similarities = self._model.wv.cosine_similarities(
                    target_vector, vectors
                )

                max_index = similarities.argmax()
                top_match = EmbedPredictionsRow(
                    pred_label=words[max_index],
                    pred_prob=float(similarities[max_index]),
                )

                res.append(top_match)

            # detpred = res
            # predictions = pd.DataFrame(detpred, columns=["pred_label", "pred_prob"])
            # txt_cols.extend(["pred_label", "base_document_number"])
            # df["pred_num"] = df["pred_label"].map({v: k for k, v in det_name_map.items()})
            # return predictions.to_json(orient="records")

            result[y] = res

            X = pd.concat(
                [
                    X,
                    pd.DataFrame(
                        {
                            f"pred_{y}": [item.pred_label for item in res],
                            f"prob_{y}": [item.pred_prob for item in res],
                        }
                    ),
                ],
                axis=1,
                ignore_index=False,
            )

        return X.to_json(
            orient="records", force_ascii=False, date_format="iso", date_unit="s"
        )

    def sentence_vector(self, words: list[str], wv):
        v = [wv[word] for word in words]
        return np.mean(v, axis=0)

    @staticmethod
    @lru_cache(maxsize=20_000)
    def wv_cached(word: str, base_name: str, model_type: str):
        model_manager = get_model_manager()
        model = model_manager.get_model(model_type, base_name, log=False)
        return model._model.wv[word]

    @staticmethod
    @lru_cache()
    def sentence_vector_cached(words: tuple[str], base_name: str, model_type: str):
        v = [FastTextModel.wv_cached(word, base_name, model_type) for word in words]
        return np.mean(v, axis=0)

    def _save_column_model(self, column, item=None):
        # TODO:
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
