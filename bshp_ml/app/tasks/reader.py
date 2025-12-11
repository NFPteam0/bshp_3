from datetime import datetime
import json
import os
from time import timezone
import zipfile

import pandas as pd
from .manager import TaskManager
from settings import TEMP_FOLDER, USE_DETAILED_LOG, DB_URL
from schemas.models import DataRow
from db import db_processor
import logging
from .utils import convert_dates_in_db_filter

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class Reader:
    async def read(self, data_filter) -> pd.DataFrame:
        if USE_DETAILED_LOG:
            logger.info("Start reading data")
        data = await db_processor.find(
            "raw_data", convert_dates_in_db_filter(data_filter), batch_size=1000
        )
        pd_data = pd.DataFrame(data)

        if USE_DETAILED_LOG:
            logger.info("Reading data. Done")
        return pd_data
