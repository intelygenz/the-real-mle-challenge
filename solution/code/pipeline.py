import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import mlflow
from mlflow.models import infer_signature

# Import custom functions
from  code.src.transformer import (
    ArrayOneHotEncoder,
    ColumnRenamer,
    ColumnSelector,
    CustomOrdinalEncoder,
    DropColumns,
    DropNan,
    DiscretizerTransformer,
    QueryFilter,
    StringToFloatTransformer,
    StringToInt,
)

# Global variables
DIR_REPO = Path.cwd().parent
DIR_DATA_RAW = Path(DIR_REPO) / "data" / "raw"
FILEPATH_DATA = DIR_DATA_RAW / "listings.csv"
FILEPATH_PLOTS = Path(DIR_REPO) / "solution" / "plots"
MODEL_PATH = DIR_REPO / "solution"/ "models"

COLUMNS = ['id', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'latitude', 'longitude', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds','amenities', 'price']
CAT_COLS = ['TV', 'Internet', 'Air conditioning', 'Kitchen', 'Heating', 'Wifi', 'Elevator', 'Breakfast'] 

MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}

FEATURE_NAMES = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']
TARGET_VARIABLE = "category"



mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

if __name__ == "__main__":
        
    print("Reading data")
    df_raw = pd.read_csv(FILEPATH_DATA)
    df_raw.head()

    
    print("Building preprocessing pipeline")
    preprocessing_pipeline = Pipeline(steps=[
        ('col_selector', ColumnSelector(COLUMNS)),
        ('bathroom_processing', StringToFloatTransformer({'bathrooms_text': 'bathrooms'})),
        ('cast_price', StringToInt('price', r"(\d+)")),
        ('filter_rows', QueryFilter("price >= 10")),
        ('drop_na', DropNan(axis=0)),
        ('bin_price', DiscretizerTransformer('price', 'category', bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])),
        ('array_to_cat', ArrayOneHotEncoder('amenities', CAT_COLS)),
        ('col_renamer_conditioning', ColumnRenamer(columns={'Air conditioning': 'Air_conditioning', 'neighbourhood_group_cleansed': 'neighbourhood'})),
        ('drop_cols', DropColumns('amenities'))
        ]
    )
    preprocessing_pipeline.set_output(transform='pandas')

    print("Building processing pipeline")

    ct =  ColumnTransformer(
        [
            ('ordinal_encoder', CustomOrdinalEncoder(categories=[list(MAP_NEIGHB.keys()), list(MAP_ROOM_TYPE.keys())], start_category=1), ["neighbourhood", "room_type"])
        ],
        remainder = "passthrough",
        verbose_feature_names_out=False
    )

    ct.set_output(transform='pandas')

    processing_pipeline = Pipeline(steps=[
        ('drop_na', DropNan(axis=0)),
        ('categorical', ct),
        ('col_selector', ColumnSelector(FEATURE_NAMES + [TARGET_VARIABLE]))
        ]
    )

    processing_pipeline.set_output(transform='pandas')

    data_pipeline = Pipeline(steps=[
        ('data_preprocessing', preprocessing_pipeline),
        ('data_processing', processing_pipeline)
    ])

    # fit the pipeline only with the training data
    print("Fitting pipeline")
    data_pipeline.fit(df_raw)

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    print("Saving col transformer")
    try:
        joblib.dump(ct, open(MODEL_PATH / "col_transformer.joblib", "wb+"))
    except FileNotFoundError:
        joblib.dump(ct, open(MODEL_PATH / "col_transformer.joblib", "wb+"))

    print("Saving preprocessing pipeline")
    try:
        joblib.dump(preprocessing_pipeline, open(MODEL_PATH / "preprocessing_pipeline.joblib", "wb+"))
    except FileNotFoundError:
        joblib.dump(preprocessing_pipeline, open(MODEL_PATH / "preprocessing_pipeline.joblib", "wb+"))

    print("Saving processing pipeline")
    try:
        joblib.dump(processing_pipeline, open(MODEL_PATH / "processing_pipeline.joblib", "wb+"))
    except FileNotFoundError:
        joblib.dump(processing_pipeline, open(MODEL_PATH / "processing_pipeline.joblib", "wb+"))

    print("Saving pipeline")
    try:
        joblib.dump(data_pipeline, open(MODEL_PATH / "pipeline.joblib", "wb+"))
    except FileNotFoundError:
        joblib.dump(data_pipeline, open(MODEL_PATH / "pipeline.joblib", "wb+"))

    print("Saving pipeline artifacts to mlflow")

    class ProcessingPipeline(mlflow.pyfunc.PythonModel):
        
        def __init__(self):
            self.whole_pipeline = None
            self.preprocessing_pipeline = None
            self.processing_pipeline = None
            self.column_transformer = None


        def load_artifacts(self, context):

            self.whole_pipeline = joblib.load(open(context.artifacts['whole_pipe'], 'rb'))
            self.preprocessing_pipeline = joblib.load(open(context.artifacts['prepro_pipe'], 'rb'))
            self.processing_pipeline = joblib.load(open(context.artifacts['proc_pipe'], 'rb'))
            self.column_transformer = joblib.load(open(context.artifacts['col_trans'], 'rb'))

        def predict(self, context, model_input):

            if self.whole_pipeline:
                return self.whole_pipeline.transform(model_input)
            else:
                raise ValueError("The model has not been loaded")
            
    class ProcessingPipeline(mlflow.pyfunc.PythonModel):

        """Class that applies the fitted processing pipeline to new data"""
        def __init__(self):
            self.column_transformer = None

        def load_context(self, context):

            self.whole_pipeline = joblib.load(context.artifacts['whole_pipe'])
            self.preprocessing_pipeline = joblib.load(context.artifacts['prepro_pipe'])
            self.processing_pipeline = joblib.load(context.artifacts['proc_pipe'])
            self.column_transformer = joblib.load(context.artifacts['col_trans'])

        def predict(self, context, model_input):

            if self.whole_pipeline:
                return self.whole_pipeline.transform(model_input)
            else:
                raise ValueError("The model has not been loaded")
            
    class MappingTransformer(mlflow.pyfunc.PythonModel):

        """Class that applies the category mapping to new data"""

        def __init__(self):
            pass

        def load_context(self, context):
            self.column_transformer = joblib.load(context.artifacts['col_trans'])

        def predict(self, context, model_input):
            
            columns_to_apply = ["neighbourhood", "room_type"]
            if self.column_transformer:
                try:
                    return self.column_transformer.transform(model_input)
                except:
                    try:
                        model_input[columns_to_apply] = self.column_transformer['ordinal_encoder'].transform(model_input[columns_to_apply])
                        return model_input
                    except:
                        raise ValueError(f"Necessary columns not present: {columns_to_apply}")
            else:
                raise ValueError("The model has not been loaded")
            
    
    print("Applying pipeline")
    df_processed = data_pipeline.transform(df_raw)
    X = df_processed[FEATURE_NAMES]
    y = df_processed[TARGET_VARIABLE]

    print("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

    print("Processed train data")
    print(f"Dataset shape: {X_train.shape}")

    print("Processed test data")
    print(f"Dataset shape: {X_train.shape}")

    print("Training Random Forest Model")
    
    print("Train model")
    clf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced', n_jobs=4)
    clf.fit(X_train, y_train)

    print("Model evaluation")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):1.4f}")

    y_proba = clf.predict_proba(X_test)
    roc_auc_score(y_test, y_proba, multi_class='ovr')
    print(f"ROC score: {roc_auc_score(y_test, y_proba, multi_class='ovr'):1.4f}")

    print("Saving model")
    try:
        joblib.dump(clf, open(MODEL_PATH / "classifier.joblib", "wb+"))
    except FileNotFoundError:
        joblib.dump(clf, open(MODEL_PATH / "classifier.joblib", "wb+"))


    mlflow.set_experiment(experiment_name="price_category_predictor")
    client = mlflow.client.MlflowClient()
    with mlflow.start_run() as run:
        
        
        print("Logging pipeline to mlflow")
        # log prod pipeline model
        
        mlflow.pyfunc.log_model(
            artifact_path="processing_pipeline",
            python_model = ProcessingPipeline(),
            registered_model_name="processing_pipeline",
            artifacts = {
                'whole_pipe': str(MODEL_PATH / "pipeline.joblib"),
                'prepro_pipe':  str(MODEL_PATH / "preprocessing_pipeline.joblib"),
                'proc_pipe':  str(MODEL_PATH / "processing_pipeline.joblib"),
                'col_trans':  str(MODEL_PATH / "col_transformer.joblib")
            },
            pip_requirements = open(DIR_REPO / 'solution' / "requirements.txt", 'r').read().split('\n'),
            code_paths = [ str(DIR_REPO / 'solution' / "code") ]
            
        )
        latest_version = client.search_registered_models(filter_string="name = 'processing_pipeline'")[0].latest_versions[0].version
        client.set_registered_model_alias('processing_pipeline', 'prod', latest_version)
        
        
        print("Logging transformer to mlflow")
        # save column transformer
        mlflow.pyfunc.log_model(
            artifact_path="transformer",
            python_model = MappingTransformer(),
            registered_model_name="mapping_transformer",
            artifacts = {
                'col_trans':  str(MODEL_PATH / "col_transformer.joblib")
            },
            pip_requirements = ['pandas', 'scikit-learn', 'numpy'],
            code_paths = [ str(DIR_REPO / 'solution' / "code")]
        )

        latest_version = client.search_registered_models(filter_string="name = 'mapping_transformer'")[0].latest_versions[0].version
        client.set_registered_model_alias('mapping_transformer', 'prod', latest_version)

        print("Logging model to mlflow")
        # log prod training model
        signature = infer_signature(X_test, y_train)
        
        mlflow.sklearn.log_model(
            clf,
            artifact_path = "artifacts",
            signature = signature,
            registered_model_name="price_category_clf",
            input_example = X_test[:1]
        )

        latest_version = client.search_registered_models(filter_string="name = 'price_category_clf'")[0].latest_versions[0].version
        client.set_registered_model_alias('price_category_clf', 'prod', latest_version)