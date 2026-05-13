import asyncio

import pandas as pd
from sentence_transformers import SentenceTransformer

from bshp_ml.app.schemas.models import EmbedPredictionsRow


class Transformer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def sentences_vector(self, sentences: list[str]) -> list:
        return self.model.encode(sentences)

    def count_sim_matrix(self, sentences: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(sentences)
        similarities = self.model.similarity(embeddings, embeddings)
        return similarities

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
        y_columns = ["cash_flow_item_name", "cash_flow_details_name", "year"]
        for y in y_columns:
            X["pred_{y}"]
            X["prob_{y}"]

        return X.to_json(
            orient="records", force_ascii=False, date_format="iso", date_unit="s"
        )
