"""
This file contains all functions related with the inference process
"""
import logging
import pickle

import numpy as np
from fastapi import HTTPException

from api.models.inference_models import InferenceRequest
from api.settings import AppSettings

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    This class encapsulate the use of the inferences
    """

    def __init__(self, settings: AppSettings) -> None:

        # Loading model
        with open(settings.model_path, "rb") as f:
            pickle_info = pickle.load(f)

        self.__clf = pickle_info[0]
        self.__settings = settings

    def inference(self, request: InferenceRequest) -> dict:
        """
        This method

        Args:
            request (InferenceRequest): request inference

        Returns:
            dict: result of the inference
        """
        # Create input
        X = np.array([self.__preprocess_request(request=request)])

        # Predict
        pred = self.__clf.predict(X)
        price_category = int(pred[0])

        return self.__settings.mapping_columns["price_category"][price_category]

    def __preprocess_request(self, request: InferenceRequest) -> list:
        """
        Extract and preprocesss, from request, the information relevant for the inference

        Args:
            request (InferenceRequest): Request sended to api

        Returns:
            list: list of values for inference
        """
        try:
            neighbourhood = self.__settings.mapping_columns["neighbourhood"][request.neighbourhood]
        except KeyError as key_exc:
            raise HTTPException(status_code=400, detail="Neighbourhood not valid")

        try:
            room_type = self.__settings.mapping_columns["room_type"][request.room_type]
        except KeyError as key_exc:
            raise HTTPException(status_code=400, detail="Room type not valid")

        room_type = self.__settings.mapping_columns["room_type"][request.room_type]
        return [neighbourhood, room_type, request.accommodates, request.bathrooms, request.bedrooms]
