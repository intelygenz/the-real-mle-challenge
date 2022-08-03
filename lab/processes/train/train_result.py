"""
This file contain the class definition for a train result
"""
from datetime import datetime


class TrainResult:
    """
    Class that encapsulate the information for train result
    """

    def __init__(
        self, accuracy: float, roc_auc_score: float, importances: dict, conf_m: list, report: dict, parameters: dict
    ) -> None:
        self.__date = datetime.now()
        self.__accuracy = accuracy
        self.__roc_auc_score = roc_auc_score
        self.__importances = importances
        self.__conf_m = conf_m
        self.__report = report
        self.__parameters = parameters

    def get_date(self) -> datetime:
        """
        Return the datetime of result creation

        Returns:
            datetime: datetime of result creation
        """
        return self.__date

    def get_dict(self) -> dict:
        """
        Convert the train result object in a dictionary

        Returns:
            dict: transformation
        """
        return {
            "accuracy": self.__accuracy,
            "roc_auc_score": self.__roc_auc_score,
            "feature importances": self.__importances,
            "confusion matrix": self.__conf_m,
            "report": self.__report,
            "model parameters": self.__parameters,
        }
