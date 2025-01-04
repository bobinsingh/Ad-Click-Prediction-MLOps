import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entities.artifact_entity import DataIngestionArtifact
from src.entities.config_entity import DataIngestionConfig
from src.exceptions import MyException
from src.logging import logging
from src.data.proj_data_handler import GetData


#Data Injestion Class

class DataIngestion:

    #For initializing data injestion config to get all data paths

    def __init__(self, data_ingestion_config: DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys) 


    #For exporting data to feature store   

    def export_data_to_feature_store(self)->DataFrame:         

        try:
            logging.info(f"Exporting data from mongodb")
            my_data = GetData()        
            df = my_data.get_collection_and_export_to_df(collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"Shape of dataframe: {df.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            fs_dir = os.path.dirname(feature_store_file_path)
            os.makedirs(fs_dir,exist_ok=True)
            df.to_csv(feature_store_file_path,index=False,header=True)
            return df
        
        except Exception as e:
            raise MyException(e, sys)
        


    #For saving the splitted data in feature store as well

    def save_splitted_data_to_feature_store(self, dataframe: DataFrame)-> None:
        
        try:
            train_data, test_data = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_data.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info(f"Exported train and test file path.")    

        except Exception as e:
            raise MyException(e, sys) 


    #To Initiate Data Injestion

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_to_feature_store()
            logging.info("Got the data from mongodb and saved to feature store")

            self.save_splitted_data_to_feature_store(dataframe)

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
          
        

