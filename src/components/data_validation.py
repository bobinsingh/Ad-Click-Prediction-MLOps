from src.logging import logging
from src.exceptions import MyException
from src.entities.config_entity import DataValidationConfig
from src.entities.artifact_entity import DataValidationArtifact, DataIngestionArtifact 
from src.utils.helpers import read_yaml_file, read_data
from src.constants import SCHEMA_FILE_PATH

import os
import sys
import json
import pandas as pd


#Data Validation Class

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys)


    #For Validating Numbers of column in ingested data
    
    def validate_column_numbers(self, dataframe:pd.DataFrame)-> bool:

        try:
            status = len(dataframe.columns.to_list()) == len(self.schema_config["columns"])
            logging.info(f"Is Number of Columns same in Ingested Data: [{status}]")
            return status
        except Exception as e:
            raise MyException(e, sys) 
            

    #For Validating Name of Cat/Num columns in Ingested Data
        
    def validate_column_presence(self, dataframe: pd.DataFrame) -> bool:
        try:
            dataframe_columns = dataframe.columns

            missing_num_columns = [column for column in self.schema_config["numerical_columns"] if column not in dataframe_columns]
            if missing_num_columns:
                logging.info(f"Missing Numerical Column: {missing_num_columns}")

            missing_cat_columns = [column for column in self.schema_config["categorical_columns"] if column not in dataframe_columns]
            if missing_cat_columns:
                logging.info(f"Missing Categorical Column: {missing_cat_columns}")

            return not (missing_num_columns or missing_cat_columns)

        except Exception as e:
            raise MyException(e, sys)
    
   
    #Run Data Validation
    def initiate_data_validation(self)-> DataValidationArtifact:

        try:
            validation_error_msg = ""

            test_data,train_data = (read_data(file_path=self.data_ingestion_artifact.test_file_path),
                                    read_data(file_path=self.data_ingestion_artifact.train_file_path))
            
            #For Validating No of Columns in Test data
            status = self.validate_column_numbers(dataframe=test_data)
            if not status:
                validation_error_msg += f"Columns are missing in testing dataframe. "
            else:
                logging.info(f"All required columns present in testing dataframe: {status}")    

            #For Validating No of Columns in Train data
            status = self.validate_column_numbers(dataframe=train_data)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All required columns present in training dataframe: {status}")    


            #For Validating presence of all columns in Test data
            status = self.validate_column_presence(dataframe=test_data)
            if not status:
                validation_error_msg += f"Columns are missing in testing dataframe. "
            else:
                logging.info(f"All categorical/int columns present in testing dataframe: {status}")


            #For Validating Presece of all columns in Train data
            status = self.validate_column_presence(dataframe=train_data)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All categorical/int columns present in training dataframe: {status}")


            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                validation_error_msg=validation_error_msg,
                validation_report_path=self.data_validation_config.data_validation_report_path
            )        

            report_dir = os.path.dirname(self.data_validation_config.data_validation_report_path)
            os.makedirs(report_dir,exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.data_validation_report_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e