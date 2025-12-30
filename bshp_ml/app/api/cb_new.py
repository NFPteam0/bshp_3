import logging
import traceback
from typing import Optional
import uuid
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    HTTPException,
    Header,
    Query,
)
import pandas as pd
from ml.models import Model
from schemas.models import DataRow, ExtDataRow, ModelTypes, EmbedPredictionsRow
from schemas.tasks import TaskResponse
from fastapi.encoders import jsonable_encoder
import json

from tasks.processing import (
    process_fitting_model,
    process_uploading_task,
    process_fitting_model_v2,
)
from tasks.__init__ import task_manager, Reader
from ml.models import ModelManager, get_model_manager
from settings import (
    USE_DETAILED_LOG,
)


logger = logging.getLogger(__name__)
# router = APIRouter(prefix="/v2", tags=["Новая cb"])
router = APIRouter()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)


async def get_token_from_header(token: Optional[str] = Header(None, alias="token")):
    return token


@router.post("/fit")
async def fit(
    background_tasks: BackgroundTasks,
    token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    base_name: str = Query(default=""),
    parameters: dict = Body(),
    model_manager: ModelManager = Depends(get_model_manager),
    fit_embeddings: bool = False,
    model_type: ModelTypes = Query(
        default=ModelTypes.catboost_txt
    ),  # ни на что не влияет, всегда эта модель
):
    if not base_name:
        base_name = "all_bases"
        logger.info("No base name provided, switching to %s", base_name)
    parameters["calculate_metrics"] = False  # TODO: forced
    # 1. Fit embeddings first
    if fit_embeddings:
        task_id = str(uuid.uuid4())
        task = await task_manager.create_task(task_id)
        logger.info(
            f"Start fitting model {ModelTypes.fstxt} on {'all_bases'}, TASK ID {task.task_id}"
        )

        try:
            await task_manager.update_task(
                task_id,
                type="FIT",
                status="PREPARE_FITTING",
                upload_progress=100,
                model_type=ModelTypes.fstxt.value,
                base_name="all_bases",
                parameters=parameters,
            )
            background_tasks.add_task(
                process_fitting_model, task_manager, model_manager, task_id
            )

        except Exception as e:
            logger.error(f"Error while fitting embeddings model: {e}")

            await task_manager.update_task(task_id, status="ERROR", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 2. Predict values with embeddings and pass it to catboost model
    try:
        # TODO: долго
        X_y = await _read_dataset({"data_filter": {"base_name": base_name}})
        if USE_DETAILED_LOG:
            logging.info("Loading columns: %s, %s", X_y.columns, X_y.shape)
        if X_y.empty:
            raise ValueError
    except Exception as e:
        print(traceback.format_exc())
        logger.error(
            f"Collection not found. Please, insure base {base_name} is loaded: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e))
    try:
        fsttext = model_manager.get_model(ModelTypes.fstxt, "all_bases")
        # модель фасттекст для всех баз одна
        Xy_embed = await fsttext.predict(X_y, set_classes=True)
        Xy_json = json.loads(Xy_embed)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Use fsttext model predictions for training catboost model,
    # categories, float_columns = fsttxt.columns

    model_type = ModelTypes.catboost_txt
    logger.info(f"Start fitting model {model_type.value}")

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)
    try:
        await task_manager.update_task(
            task_id,
            type="FIT",
            status="PREPARE_FITTING",
            upload_progress=100,
            model_type=model_type.value,
            base_name=base_name,
            parameters=parameters,
        )

        # Start background task
        background_tasks.add_task(
            process_fitting_model_v2, task_manager, model_manager, task_id, Xy_json
        )

        return TaskResponse(task_id=task_id, message="Task processing started")

    except Exception as e:
        logger.error(f"Error in fitting model: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(
    X: list[ExtDataRow],
    base_name: str = Query(default=""),
    # token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    model_manager: ModelManager = Depends(get_model_manager),
    model_type: ModelTypes = Query(
        default=ModelTypes.catboost_txt
    ),  # ни на что не влияет, всегда эта модель
):
    if not base_name:
        base_name = "all_bases"
    # 1. Predict values with embeddings and pass it to catboost model
    try:
        fsttext = model_manager.get_model(ModelTypes.fstxt, "all_bases")
        Xy_embed = await fsttext.predict(jsonable_encoder(X))
        Xy_json = json.loads(Xy_embed)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    logger.info("Shape of json for predictions: %s", str(pd.DataFrame(Xy_json).shape))
    model_type = ModelTypes.catboost_txt
    try:
        model = model_manager.get_model(model_type, base_name)
        result = []
        X_y_list = await model.predict(Xy_json)
        result = X_y_list
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return result


async def _read_dataset(parameters: dict) -> pd.DataFrame:
    data_filter = parameters["data_filter"]
    if USE_DETAILED_LOG:
        logger.info("Reading data from db: %s", data_filter)
    reader = Reader()
    X_y = await reader.read(data_filter)

    return X_y
