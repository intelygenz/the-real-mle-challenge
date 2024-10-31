import os
import sys
from pathlib import Path
import logging
import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DIR_REPO = Path(__file__).parent.parent.parent.parent.parent
os.chdir(DIR_REPO)

# Import custom functions
from code.src.transformer import (
    ColumnSelector,
    CustomOrdinalEncoder,
    DropNan
)

LOG_DIR =  DIR_REPO / 'solution' / 'logs' 
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "test_explore.log")

# Configure logging
logging.basicConfig(
    filename=log_file, 
    level=logging.DEBUG,
    filemode='w+'
)

logger = logging.getLogger(__name__)


DIR_DATA_PROCESSED = Path(DIR_REPO) / "data" / "processed"
DIR_MODELS = Path(DIR_REPO) / "models"
FILEPATH_PROCESSED = DIR_DATA_PROCESSED / "preprocessed_listings.csv"

MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}
FEATURE_NAMES = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']
TARGET_VARIABLE = "category"

df = pd.read_csv(FILEPATH_PROCESSED, index_col=0)

logging.info("Aplying old code")
df_old = df.copy()

t1_old_code = datetime.datetime.now()

df_old = df_old.dropna(axis=0)
# Map categorical features
df_old["neighbourhood"] = df_old["neighbourhood"].map(MAP_NEIGHB)
df_old["room_type"] = df_old["room_type"].map(MAP_ROOM_TYPE)
X_old = df_old[FEATURE_NAMES]
y_old = df_old['category']


X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.15, random_state=1)

clf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced', n_jobs=4)
clf.fit(X_train_old, y_train_old)

y_pred_old = clf.predict(X_test_old)

acc_old = accuracy_score(y_test_old, y_pred_old)

y_proba_old = clf.predict_proba(X_test_old)
roc_old = roc_auc_score(y_test_old, y_proba_old, multi_class='ovr')

maps = {'0.0': 'low', '1.0': 'mid', '2.0': 'high', '3.0': 'lux'}

report = classification_report(y_test_old, y_pred_old, output_dict=True)
df_report = pd.DataFrame.from_dict(report).T[:-3]
df_report.index = [maps[i] for i in df_report.index]
df_report_old = df_report.copy()

t2_old_code = datetime.datetime.now()

logging.info("Aplying new code")
t1_new_code = datetime.datetime.now()


ct =  ColumnTransformer(
    [
        ('ordinal_encoder', CustomOrdinalEncoder(categories=[list(MAP_NEIGHB.keys()), list(MAP_ROOM_TYPE.keys())], start_category=1), ["neighbourhood", "room_type"])
    ],
    remainder = "passthrough",
    verbose_feature_names_out=False
)

processing_pipeline = Pipeline(steps=[
    ('drop_na', DropNan(axis=0)),
    ('categorical', ct),
    ('col_selector', ColumnSelector(FEATURE_NAMES + [TARGET_VARIABLE]))
    ]
)

processing_pipeline.set_output(transform='pandas')

df_processed = processing_pipeline.fit_transform(df)

X_new = df_processed[FEATURE_NAMES]
y_new = df_processed['category']


X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.15, random_state=1)

clf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced', n_jobs=4)
clf.fit(X_train_new, y_train_new)

y_pred_new = clf.predict(X_test_new)

acc_new = accuracy_score(y_test_new, y_pred_new)

y_proba_new = clf.predict_proba(X_test_new)
roc_new = roc_auc_score(y_test_new, y_proba_new, multi_class='ovr')

maps = {'0.0': 'low', '1.0': 'mid', '2.0': 'high', '3.0': 'lux'}

report = classification_report(y_test_old, y_pred_old, output_dict=True)
df_report = pd.DataFrame.from_dict(report).T[:-3]
df_report.index = [maps[i] for i in df_report.index]
df_report_new = df_report.copy()

t2_new_code = datetime.datetime.now()

df_old = pd.concat([X_old, y_old], axis=1)


logging.info(f"""
Old code time: {t2_old_code - t1_old_code}
New code time: {t2_new_code - t1_new_code}
Same accuracy: {acc_old == acc_new}
Same roc: {roc_old == roc_new}
Same result: {all(df_processed == df_old)}
Same report: {all(df_report_new == df_report_old)}
""")