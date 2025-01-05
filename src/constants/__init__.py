import os
from datetime import date


PIPELINE_NAME: str = ""
TARGET_COLUMN = "click"
CURRENT_YEAR = date.today().year


#Data Ingestion related constants
DATA_INGESTION_COLLECTION_NAME: str = "Ad_click_proj_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20

ARTIFACT_DIR: str = "artifact"
INGESTED_FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


#Data Validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"
SCHEMA_FILE_PATH = os.path.join("configs", "schema.yaml")


#Data Transformation related constants
DATA_TRANFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_PREPROCESSING_OBJECT_DIR: str = "transformed_object"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
IMPUTE_KNN_N_NEIGHBOURS: int = 5



