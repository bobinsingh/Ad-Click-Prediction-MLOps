import sys
from src.entities.config_entity import AdPredictorConfig
from src.entities.s3_config import CloudModelEstimator
from src.exceptions import MyException
from src.logging import logging
from pandas import DataFrame


class AdData:
    def __init__(self,
                gender_Male,
                gender_Non_Binary,
                age,
                device_type_Mobile,
                device_type_Tablet,
                ad_position_Side,
                ad_position_Top,
                browsing_history_Entertainment,
                browsing_history_News,
                browsing_history_Shopping,
                browsing_history_Social_Media,
                time_of_day_Evening,
                time_of_day_Morning,
                time_of_day_Night):
        """
        Ad Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.gender_Male = gender_Male
            self.gender_Non_Binary = gender_Non_Binary
            self.age = age
            self.device_type_Mobile = device_type_Mobile
            self.device_type_Tablet = device_type_Tablet
            self.ad_position_Side = ad_position_Side
            self.ad_position_Top = ad_position_Top
            self.browsing_history_Entertainment = browsing_history_Entertainment
            self.browsing_history_News = browsing_history_News
            self.browsing_history_Shopping = browsing_history_Shopping
            self.browsing_history_Social_Media = browsing_history_Social_Media
            self.time_of_day_Evening = time_of_day_Evening
            self.time_of_day_Morning = time_of_day_Morning
            self.time_of_day_Night = time_of_day_Night

        except Exception as e:
            raise MyException(e, sys) from e
        

    def get_ad_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from AdData class input
        """
        try:
            logging.info("Entered Get Ad Input Dataframe function")
            ad_input_dict = self.get_ad_data_as_dict()
            df = DataFrame(ad_input_dict)
            logging.info("Input Columns Recieved: ",df.columns.to_list)
            logging.info("Exited Get Ad Input Dataframe function")
            return df
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_ad_data_as_dict(self):
        """
        This function returns a dictionary from AdData class input
        """
        logging.info("Entered get_ad_data_as_dict method as AdData class")

        try:
            input_data = {
                    "gender_Male": [self.gender_Male],
                    "gender_Non-Binary": [self.gender_Non_Binary],
                    "age": [self.age],
                    "device_type_Mobile": [self.device_type_Mobile],
                    "device_type_Tablet": [self.device_type_Tablet],
                    "ad_position_Side": [self.ad_position_Side],
                    "ad_position_Top": [self.ad_position_Top],
                    "browsing_history_Entertainment": [self.browsing_history_Entertainment],
                    "browsing_history_News": [self.browsing_history_News],
                    "browsing_history_Shopping": [self.browsing_history_Shopping],
                    "browsing_history_Social Media": [self.browsing_history_Social_Media],
                    "time_of_day_Evening": [self.time_of_day_Evening],
                    "time_of_day_Morning": [self.time_of_day_Morning],
                    "time_of_day_Night": [self.time_of_day_Night]
                }

            logging.info("Created Ad data dict")
            logging.info("Exited get_ad_data_as_dict method as AdData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e



class AdDataClassifier:
    def __init__(self, prediction_pipeline_config: AdPredictorConfig = AdPredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)
        

    def predict(self, dataframe) -> str:
        """
        This is the method of AdDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of AdDataClassifier class")
            model = CloudModelEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)
