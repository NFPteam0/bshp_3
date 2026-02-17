import gc
import pandas as pd
from sklearn.pipeline import Pipeline
from ml.data_processing import NanProcessor
from ml.fstxt.utils import prepare_sentences
from schemas.models import ExtEmbedPredictionsRow, ModelTypes

from ml.data_processing import Checker
from .model import BATCH_SIZE, FastTextModel
from .model import logger
from settings import (
    USE_DETAILED_LOG,
)


class ExtFastTextModel(FastTextModel):
    model_type = ModelTypes.extfstxt

    def _sync_predict(
        self,
        X_api: dict | list[dict],
        for_metrics=False,
        set_classes: bool = False,
        set_from: dict = None,
    ) -> dict[str, ExtEmbedPredictionsRow]:
        X = pd.DataFrame(X_api)

        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")

        for col in ["cash_flow_item_name", "cash_flow_details_name"]:
            if col not in X.columns:
                X[col] = ""

        # 1. Validating
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

        # 2. Make set of prediction classes
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
                        set_from[self.name2code[col]].astype(int),
                    )
                )
                for col in self.y_columns
            }
        if self.all_classes_names is None or self.all_classes_codes is None:
            raise ValueError(f"Model is not ready, it's {self.status}. Fit it before.")

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

        PP_UNFEATURED = UNFEATURED + [
            "article_name",
            "analytic",
            "analytic2",
            "analytic3",
            "article_parent",
            "article_group",
            "article_kind",
        ]

        # 3. Preprocess sentences
        sentences = prepare_sentences(
            X,
            [
                col
                for col in self.str_columns
                if col not in UNFEATURED + ["payment_purpose"]
                and col not in self.y_columns
            ],
        )

        sentences_pp = prepare_sentences(
            X,
            [
                col
                for col in self.str_columns
                if col not in PP_UNFEATURED and col not in self.y_columns
            ],
        )

        # 4. Predict

        for y in self.y_columns:
            class_matrix, class_names = self.build_class_matrix(
                self.all_classes_names[y], self.all_classes_codes[y]
            )

            preds, probs = self.batched_predict(
                sentences, class_matrix, class_names, BATCH_SIZE
            )

            preds_pp, probs_pp = self.batched_predict(
                sentences_pp, class_matrix, class_names, BATCH_SIZE
            )

            X[f"pred_{y}"] = preds
            X[f"prob_{y}"] = probs

            X[f"pred_pp_{y}"] = preds_pp
            X[f"prob_pp_{y}"] = probs_pp

            gc.collect()

        return X.to_json(
            orient="records", force_ascii=False, date_format="iso", date_unit="s"
        )
