import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DIR_REPO = Path.cwd()

DIR_DATA_PROCESSED = DIR_REPO / "data" / "processed"

DIR_MODELS = DIR_REPO / "models"
FILEPATH_MODEL = DIR_MODELS / "ny_classifier.pkl"

N_ESTIMATORS = 500
SEED = 0
CLASS_WEIGHT = "balanced"
N_JOBS = 4


# Load data

data_x = pd.read_parquet(DIR_DATA_PROCESSED / "ny_listings_x.parquet")
data_y = pd.read_parquet(DIR_DATA_PROCESSED / "ny_listings_y.parquet").iloc[:, 0]


# Train model

classifier = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=SEED,
    class_weight=CLASS_WEIGHT,
    n_jobs=N_JOBS,
)
classifier.fit(data_x, data_y)


# Save model

with open(FILEPATH_MODEL, "wb") as model_file:
    pickle.dump(classifier, model_file)
