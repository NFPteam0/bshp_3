import csv
import logging

from pydantic import BaseModel
from settings import settings

logger = logging.getLogger(__name__)

METRICS_CSV_PATH = settings.METRICS_FOLDER / "metrics.csv"


class MetricsTrain(BaseModel):
    model_name: str  # modeltype
    dataset_name: str  # probably parametrs.data_filter
    accuracy_year: float
    accuracy_item: float
    accuracy_details: float
    time: float  # time for training
    time_start: str  # time when training started, ISO format
    time_end: str  # time when training ended, ISO format
    data_size: int  # number of rows in dataset


# class MetricsPredict(BaseModel):
#     model_name: str
#     dataset_name: str
#     accuracy: float
#     time: float  # time for testing
#     time_start: str  # time when testing started, ISO format
#     time_end: str  # time when testing ended, ISO format
#     ram_usage: float
#     disk_usage: float  # catboost model size on disk, MB
#     data_size: int  # number of rows in dataset


def write_in_csv(metrics: MetricsTrain) -> None:
    """
    Appends metrics to metrics.csv file. Creates file if it doesn't exist.
    Only appends rows, never overwrites or deletes.
    """
    settings.METRICS_FOLDER.mkdir(parents=True, exist_ok=True)

    file_exists = METRICS_CSV_PATH.exists()

    try:
        with open(METRICS_CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(MetricsTrain.model_fields.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if file is new
            if not file_exists:
                writer.writeheader()

            # Write metrics row
            writer.writerow(metrics.model_dump())
            logger.info(
                f"Metrics saved to CSV: {metrics.model_name} | "
                f"year={metrics.accuracy_year:.4f} "
                f"item={metrics.accuracy_item:.4f} "
                f"details={metrics.accuracy_details:.4f} "
                f"time={metrics.time:.1f}s rows={metrics.data_size}"
            )

    except Exception as e:
        logger.error(f"Failed to write metrics to CSV: {e}")
        raise


# def test_metrics(metrics: MetricsTrain):
#     '''
#     Clipped Json with test data -> ...
#     '''
#     pass
