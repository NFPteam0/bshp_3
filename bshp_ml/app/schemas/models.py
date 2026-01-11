import pandas as pd
from pydantic import BaseModel, field_validator
from enum import Enum
from datetime import datetime

from typing import Optional, Dict


class DataRow(BaseModel):
    """
    Loading, input or output data row
    """

    # TODO: разбить на отдельные pydantic-и

    number: str
    date: datetime
    is_reverse: bool
    moving_type: str
    kind: str
    company_inn: str
    company_kpp: str
    base_document_number: str | int
    base_document_date: datetime
    base_document_kind: str
    base_document_operation_type: str
    contractor_name: str
    contractor_inn: str
    contractor_kpp: str
    contractor_kind: str
    article_name: str
    article_code: str
    is_main_asset: bool
    analytic: str
    analytic2: str
    analytic3: str
    article_document_number: str | int
    article_document_date: datetime
    article_parent: str
    article_group: str
    article_kind: str
    row_number: int | int
    article_row_number: int
    store: str
    department: str
    company_account_number: str
    contractor_account_number: str
    qty: float
    price: float
    sum: float
    cash_flow_item_code: str
    cash_flow_details_code: str
    year: str

    @field_validator("date", mode="before")
    def check_date(cls, value):
        if isinstance(value, str):
            result = datetime.strptime(value, r"%d.%m.%Y %H:%M:%S")
        else:
            result = value

        return result

    @field_validator("base_document_date", mode="before")
    def check_base_document(cls, value):
        if isinstance(value, str):
            result = datetime.strptime(value, r"%d.%m.%Y %H:%M:%S")
        else:
            result = value

        return result

    @field_validator("article_row_number", mode="before")
    def check_article_row_number(cls, value):
        if not value:
            return 0
        elif isinstance(value, str):
            result = int(value)
        elif isinstance(value, int):
            result = value
        else:
            return 0
        return result

    @field_validator("article_document_date", mode="before")
    def check_article_document_date(cls, value):
        if not value:
            return datetime(1, 1, 1)
        elif isinstance(value, str):
            result = datetime.strptime(value, r"%d.%m.%Y %H:%M:%S")
        else:
            result = value

        return result


class ExtDataRow(DataRow):
    cash_flow_details_name: str | None = None
    cash_flow_item_name: str | None = None
    payment_purpose: str | None = None
    payment_purpose_returned: str | None = None


class MetadataCb(Enum):
    x: tuple[str] = ""
    y: tuple[str] = ""
    txt: tuple[str] = ""
    cat: tuple[str] = ""


class EmbedPredictionsRow(BaseModel):
    pred_label: str
    pred_prob: float


class ModelStatuses(Enum):
    CREATED = "CREATED"
    FITTING = "FITTING"
    READY = "READY"
    ERROR = "ERROR"


class ModelInfo(BaseModel):
    status: ModelStatuses
    error_text: str
    fitting_start_date: Optional[datetime]
    fitting_end_date: Optional[datetime]
    metrics: Optional[Dict[str, float]]


class ModelTypes(str, Enum):
    rf = "rf"
    catboost = "catboost"
    fstxt = "fasttext"
    catboost_txt = "catboost+"
