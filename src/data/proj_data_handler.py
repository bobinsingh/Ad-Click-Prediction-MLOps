import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.config.mongo_db_handler import MongoDBClient
from src.exceptions import MyException
from src.config.mongo_db_config import DATABASE_NAME
from src.logging import logging


# Class for Initializing connection with MongoDB and Retrieving Proj1-Data

class GetData:

    #Initializing connection with DB

    def __init__(self) -> None:
        
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)


    def get_collection_and_export_to_df(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:

        #Function for retrieving data from collection in the form of key value pairs and export it into the form of a pd-Dataframe

        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            #Next steps are for converting Key:Value pair collection data to a pandas dataframe and export it with some preprocessing.

            logging.info("Data Retrieved, Converting to DataFrame")

            df = pd.DataFrame(list(collection.find()))
            print(f"Data fecthed with len: {len(df)}")
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df

        except Exception as e:
            raise MyException(e, sys)
        