"""
This file have functions for the train of the model
"""
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from processes.config import ConfigTrain
from processes.train.train_result import TrainResult


def train(df: pd.DataFrame):
    """This function train and save results

    Args:
        df (pd.DataFrame): preprocessed dataframe
    """

    X, y = df[ConfigTrain.FEATURE_NAMES], df[ConfigTrain.FEATURE_CATEGORY]

    # Division train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ConfigTrain.TEST_SIZE, random_state=ConfigTrain.RANDOM_STATE_SPLIT
    )

    # Create and train model
    clf = RandomForestClassifier(
        n_estimators=ConfigTrain.N_ESTIMATORS,
        random_state=ConfigTrain.RANDOM_STATE_TRAIN,
        class_weight=ConfigTrain.CLASS_WEIGHT,
        n_jobs=ConfigTrain.N_JOBS,
    )
    clf.fit(X_train, y_train)

    # Create result model
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    train_result = TrainResult(
        accuracy=accuracy_score(y_test, y_pred),
        roc_auc_score=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        importances=dict(zip(ConfigTrain.FEATURE_NAMES, clf.feature_importances_)),
        conf_m=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred, output_dict=True),
        parameters={
            "n_estimators": ConfigTrain.N_ESTIMATORS,
            "class_weight": ConfigTrain.CLASS_WEIGHT,
            "test_split": ConfigTrain.TEST_SIZE,
        },
    )

    # Save model and result
    model_name = ConfigTrain.FOLDER_PATH + "random_forest_classifier_" + str(train_result.get_date()) + ".pkl"
    pickle.dump((clf, train_result), open(model_name, "wb"))
