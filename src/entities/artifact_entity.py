from dataclasses import dataclass


# For Data Ingestion
@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str