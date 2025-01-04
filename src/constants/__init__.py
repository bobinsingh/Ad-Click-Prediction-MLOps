import os
from datetime import date


PIPELINE_NAME: str = ""


#Data Ingestion related constant

DATA_INGESTION_COLLECTION_NAME: str = "Ad_click_proj_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20

ARTIFACT_DIR: str = "artifact"
INGESTED_FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"









SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")




