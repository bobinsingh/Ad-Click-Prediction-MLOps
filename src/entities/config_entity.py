import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


#Training Pipeline Configs
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


#Data Ingestion Component Configs
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, INGESTED_FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    split_random_state: int = SPLIT_RANDOM_STATE
    collection_name:str = DATA_INGESTION_COLLECTION_NAME


#Data Validation Component Configs
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    data_validation_report_path: str = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)


#Data Transformation Component Configs
@dataclass
class DataTransformationConfig:
    data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_PREPROCESSING_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)
    knn_n_neighbours: int = IMPUTE_KNN_N_NEIGHBOURS


#Model Trainer Component Configs
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
    trained_model_parameters_path: str = os.path.join(model_trainer_dir, TRAINED_MODEL_DIR,TRAINED_MODEL_PARAMETERS)
    trained_model_metrics_path: str = os.path.join(model_trainer_dir, TRAINED_MODEL_DIR,TRAINED_MODEL_METRICS)
    model_config_file_path: str = MODEL_HYPERPARAMETERS_FILE_PATH


#Model Evaluation Component Configs
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = S3_STORED_MODEL_FILE_NAME


#Model Pusher Component Configs
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = S3_STORED_MODEL_FILE_NAME  


@dataclass
class AdPredictorConfig:
    model_file_path: str = TRAINED_MODEL_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME    