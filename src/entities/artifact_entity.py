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