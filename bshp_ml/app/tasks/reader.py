import logging

import pandas as pd
from settings import USE_DETAILED_LOG

from db import db_processor

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class Reader:
    async def read(self, data_filter) -> pd.DataFrame:
        if USE_DETAILED_LOG:
            logger.info("Start reading data")
        # почему? data = await db_processor.find(
        #     "raw_data", convert_dates_in_db_filter(data_filter), batch_size=1000
        data = await db_processor.find("raw_data", data_filter, batch_size=1000)
        pd_data = pd.DataFrame(data)

        if USE_DETAILED_LOG:
            logger.info("Reading data. Done")
        return pd_data

    async def read_limited(self, data_filter, limit=80_000) -> pd.DataFrame:
        if USE_DETAILED_LOG:
            logger.info("Start reading data")
        # почему? data = await db_processor.find(
        #     "raw_data", convert_dates_in_db_filter(data_filter), batch_size=1000
        data = await db_processor.find_limited(
            "raw_data", data_filter, limit=limit, batch_size=1000
        )
        pd_data = pd.DataFrame(data)

        if USE_DETAILED_LOG:
            logger.info("Reading data. Done")
        return pd_data
