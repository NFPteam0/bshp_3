import json
import os
from datetime import datetime, timezone
import zipfile

import pandas as pd
from .manager import TaskManager
from settings import TEMP_FOLDER, USE_DETAILED_LOG, DB_URL
from schemas.models import DataRow, ExtDataRow
from db import db_processor
import logging

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class DataLoader:
    """Это доступ к данным + сервисный"""

    # TODO: переделать
    def __init__(self):
        pass

    async def upload_data_from_file(self, task_manager: TaskManager, task):
        if USE_DETAILED_LOG:
            logger.info("saving data to temp zip file")

        await task_manager.update_task(
            task.task_id, status="UNZIPPING _DATA", progress=10
        )
        folder = os.path.join(TEMP_FOLDER, task.task_id)
        os.makedirs(folder)
        if USE_DETAILED_LOG:
            logger.info("reading  data from zip file, unzipping")
        await self.get_data_from_zipfile(task.file_path, folder)

        zip_filename = os.path.basename(task.file_path)
        zip_filename_without_ext = os.path.splitext(zip_filename)[0]
        data_file_path = os.path.join(folder, f"{zip_filename_without_ext}.json")

        with open(data_file_path, "r", encoding="utf-8-sig") as fp:
            json_data = json.load(fp)
        if USE_DETAILED_LOG:
            logger.info("validatind uploaded data")
        await task_manager.update_task(
            task.task_id, status="VALIDATING _DATA", progress=20
        )

        data = []
        for row in json_data:
            data_row = ExtDataRow.model_validate(row).model_dump()
            data.append(data_row)

        pd_data = pd.DataFrame(data)
        pd_data["base_name"] = task.base_name
        pd_data["uploading_date"] = datetime.now(tz=timezone.utc)

        data = pd_data.to_dict(orient="records")
        if USE_DETAILED_LOG:
            logger.info("writing data to db")
        await task_manager.update_task(
            task.task_id, status="WRITING _TO_DB", progress=60
        )

        if task.replace:
            await db_processor.delete_many("raw_data")

        await db_processor.insert_many("raw_data", data)

        await task_manager.cleanup_task_files(task.task_id)

        return data

    async def get_data_from_zipfile(self, zip_file_path, folder):
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(folder)

    async def delete_data(self, db_filter=None):
        await db_processor.delete_many("raw_data", db_filter=db_filter)

    async def get_data_count(self, accounting_db="", db_filter=None):
        result = await db_processor.get_count("raw_data", db_filter=db_filter)
        return result

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
