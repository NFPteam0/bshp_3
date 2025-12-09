import json
import logging
import os

from sklearn.pipeline import Pipeline
from .utils import prepare_sentences, preprocess_text
import numpy as np
from gensim.models import FastText
from ..models import Model
from ml.data_processing import (
    Checker,
    DataEncoder,
    FeatureAdder,
    NanProcessor,
    Shuffler,
)
from schemas.models import ModelStatuses, ModelTypes
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
    status = None
    _model: FastText = None

    def __init__(self, base_name: str):
        super().__init__(base_name)
        self.str_columns.extend(["cash_flow_item_name", "cash_flow_details_name"])

    async def load(self, uid):
        self.uid = uid

        if self.status != ModelStatuses.ERROR:
            # for y_col in self.y_columns:
            #     if y_col == "cash_flow_details_code":
            #         for item in self.classes[y_col]:
            #             self._load_column_model(y_col, item)
            #     else:
            #         self._load_column_model(y_col)
            # self._load_column_model(y_col)

            self._load_encoder()

    def _load_pretrained(self, model_folder=MODEL_FOLDER):
        self._model = FastText.load(f"{model_folder}/pretrained/fsttxt.model")

    def _fit(self, df: pd.DataFrame, parameters, is_first=True):
        df[self.str_columns]
        if USE_DETAILED_LOG:
            logger.info("{} fit".format("First" if is_first else "continuous"))

        if is_first:
            self.strict_acc = {}
            self.test_strict_acc = {}
        c_x_columns = self.x_columns.copy()

        indexes_to_encode = []
        for ind, col in enumerate(self.x_columns):
            if col in self.columns_to_encode:
                indexes_to_encode.append(ind)

    def _get_pipeline(self) -> Pipeline:
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

        return Pipeline(pipeline_list)

    def fit_continue(self):
        if self._model is None:
            self._load_pretrained()
        sentences = prepare_sentences(X, self.str_columns)
        new_sentences = [preprocess_text(" ".join(sent)).split() for sent in sentences]
        self._model.build_vocab(new_sentences, update=True)
        self._model.train(
            new_sentences,
            total_examples=len(new_sentences),
            epochs=self._model.epochs,
        )

    def _create(self):
        self._model = FastText(
            sentences=prep_sents,
            vector_size=2**7,
            window=30,
            min_count=1,
            workers=4,
            sg=1,
            min_n=2,
            max_n=8,
            epochs=15,
        )

    def predict(self, X, for_metrics=False):
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError("Model is not ready. Fit it before.")

        field_models = self.field_models
        X = pd.DataFrame(X)
        row_numbers = list(X.index)

        X[self.y_cols] = ""

        pipeline = self._get_pipeline()
        X_y = pipeline.transform(X).copy()
        X_result = X.copy()

        X = X_y[self.x_columns].to_numpy()
        y = model.predict(X)
        X_y[y_col] = y.ravel()
        # details cols
        all_classes_names = df["detail_name"]
        wordvec = dict(zip(all_classes_names, [self._model.wv[cls] for cls in c]))
        res = []
        preprocessed_sentences = []

        for sentence in sentences:
            # Предобработка текста
            processed_text = preprocess_text(" ".join(sentence[:-1]))
            preprocessed_sentences.append(processed_text)

            # Получение вектора и вычисление схожести
            target_vector = self._model.model.wv[processed_text]

            words = list(wordvec.keys())
            vectors = list(wordvec.values())

            # Вычисление косинусной схожести и поиск максимума за один проход
            similarities = self._model.model.wv.cosine_similarities(
                target_vector, vectors
            )
            max_index = similarities.argmax()
            top_match = (words[max_index], similarities[max_index])

            res.append(top_match)

        detpred, sents = get_details(sentences)
        predictions = pd.DataFrame(detpred, columns=["pred_label", "pred_prob"])
        txt_cols.extend(["pred_label", "base_document_number"])

        df["pred_num"] = df["pred_label"].map({v: k for k, v in det_name_map.items()})
        return res, preprocessed_sentences

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
