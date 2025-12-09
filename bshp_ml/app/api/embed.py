import logging
import uuid
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query
from bshp_ml.app.schemas.models import ModelTypes
from bshp_ml.app.schemas.tasks import TaskResponse

from tasks import task_manager

router = APIRouter(
    prefix="/embeddings", tags=["Embedding модель (по умолчанию FastText)"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

logger = logging.getLogger(__name__)


@router.get("/fit")
async def fit_embeddings(
    background_tasks: BackgroundTasks,
    # token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    base_name: str = Query(default=""),
    model_type: ModelTypes = Query(default=ModelTypes.fstxt),
    parameters: dict = Body(),
) -> TaskResponse:
    logger.info(f"Start fitting model")

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

    try:
        if not base_name:
            base_name = "all_bases"
            base_name = "all_bases"
        # Update task
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
        background_tasks.add_task(process_fitting_model, task_id)

        return TaskResponse(task_id=task_id, message="Task processing started")

    except Exception as e:
        logger.error(f"Error in fitting model: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict")
async def fit_embeddings(
    background_tasks: BackgroundTasks,
    # token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    base_name: str = Query(default=""),
    model_type: ModelTypes = Query(default=ModelTypes.rf),
    parameters: dict = Body(),
) -> TaskResponse:
    logger.info(f"Start fitting model")

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

    try:
        if not base_name:
            base_name = "all_bases"
            base_name = "all_bases"
        # Update task
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
        background_tasks.add_task(process_fitting_model, task_id)

        return TaskResponse(task_id=task_id, message="Task processing started")

    except Exception as e:
        logger.error(f"Error in fitting model: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
