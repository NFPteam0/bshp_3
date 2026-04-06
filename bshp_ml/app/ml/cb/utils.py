import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def eval_model(true_labels, predictions) -> tuple:
    ac = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    # print(f"Accuracy: {ac:.4f}, F1: {f1:.4f}")
    return ac, f1


def make_all_data(df: pd.DataFrame, y: str):
    """
    Создает копию df длинной len(df[y].unique()), т.е.
    общий срез датасета, где каждый объект искомого
    класса - уникальный
    """

    all_data = df.drop_duplicates(subset=[y], keep="first").copy()
    return all_data


def encode_cat(col: pd.Series, encoder: dict = None):
    if encoder:
        return col.map(encoder), encoder
    else:
        encoder = {name: num for num, name in enumerate(col.unique())}
        return col.map(encoder), encoder


def decode_cat(col: pd.Series, decoder: dict = None):
    if decoder:
        return col.map(decoder), decoder
    else:
        decoder = {name: num for num, name in enumerate(col.unique())}
        return col.map(decoder), decoder


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


def add_data(df: pd.DataFrame, sample_frac: float = 0.05) -> pd.DataFrame:
    """
    Добавить в датасет сгенерированные данные для повышения веса pred_* полей.
    Создаёт примеры с минимальной информацией (только pred_* и финансовые данные),
    чтобы модель лучше училась на случаях, когда есть только предсказания текстовой модели.

    Args:
        df: исходный датасет
        sample_frac: доля от исходного df для создания синтетических примеров (по умолчанию 0.1)

    Гарантирует наличие всех уникальных cash_flow_details_code и cash_flow_item_code.
    Отбираются строки, где обнуляются поля, не связанные с платежом и его классификацией.
    """

    columns_to_clear = [
        "contractor_name",
        "contractor_kpp",
        "company_inn",
        "contractor_inn",
        "contractor_kind",
        "base_document_kind",
        "article_parent",
        "article_code",
        "article_document_number",
        "base_name",
        "base_document_number",
        "payment_purpose_returned",
        "kind",
        "contract_name",
        "contract_number",
        "accepted_issued",
    ]

    synthetic_parts = []

    # 1. Один пример для каждого cash_flow_details_code
    if "cash_flow_details_code" in df.columns:
        details_data = df.drop_duplicates(
            subset=["cash_flow_details_code"], keep="first"
        )
        synthetic_parts.append(details_data)

    # 2. Один пример для каждого cash_flow_item_code
    if "cash_flow_item_code" in df.columns:
        item_data = df.drop_duplicates(subset=["cash_flow_item_code"], keep="first")
        synthetic_parts.append(item_data)

    # 3. Случайная выборка для разнообразия
    sample_size = max(1, int(len(df) * sample_frac))
    random_data = df.sample(n=sample_size, random_state=42).copy()
    synthetic_parts.append(random_data)

    # Объединяем и удаляем дубликаты
    synthetic_data = pd.concat(synthetic_parts, ignore_index=True)
    for col in columns_to_clear:
        if col in synthetic_data.columns:
            if synthetic_data[col].dtype == "object":
                synthetic_data[col] = ""
            else:
                synthetic_data[col] = -1

    df = pd.concat([df, synthetic_data], ignore_index=True)
    return df
