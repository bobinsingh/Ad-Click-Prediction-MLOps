from dataclasses import dataclass


# For Data Ingestion
@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str


# For Data Validation
@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_error_msg: str
    validation_report_path: str


# For Data Transformation
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str


#For Classification Metrics
@dataclass
class ClassificationMetricArtifact:
    accuracy:float
    f1_score:float
    precision_score:float
    recall_score:float


#For Model Trainer
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    trained_model_metrics_path:str
    trained_model_parameters_path:str 
    metric_artifact:ClassificationMetricArtifact      


#For Model Evaluation
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str


#For Model Pusher
@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str