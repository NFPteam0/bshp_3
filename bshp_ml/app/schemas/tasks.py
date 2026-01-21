from pydantic import BaseModel, field_validator
from enum import Enum
from datetime import datetime

from typing import Optional, Dict


class TaskData(BaseModel):
    task_id: str
    type: str = "FIT"
    status: str = "CREATED"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    base_name: Optional[str] = None
    replace: Optional[bool] = None
    model_type: Optional[str] = None
    parameters: Optional[dict] = None

    # Внутренние поля
    file_path: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    message: str


class StatusResponse(BaseModel):
    status: str
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    description: Optional[str] = None


class ProcessingTaskResponse(BaseModel):
    task_id: str
    type: str
    base_name: str
    status: str
