"""
This file contains the definition of the api, as simple as possible, to make an inference from model
"""

from fastapi import FastAPI

from api.inference_engine.inference_engine import InferenceEngine
from api.models.inference_models import InferenceRequest, InferenceResponse
from api.settings import AppSettings

settings = AppSettings()
inference_engine = InferenceEngine(settings=settings)
app = FastAPI()


@app.get("/")
async def root():
    """
    Root endpoint

    Returns:
        dict: default message for api
    """
    return {"message": "Welcome to " + settings.app_name}


@app.get("/inference/")
async def inference(request: InferenceRequest) -> InferenceResponse:
    """
    This endpoint recevie an inference request and return the result of them

    Args:
        request (InferenceRequest): Inference received

    Returns:
        InferenceResponse: Inference response created
    """
    return InferenceResponse(id=request.id, price_category=inference_engine.inference(request))
