import sys
import os
import json
from typing import Tuple

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exceptions import MyException
from src.logging import logging
from src.constants import MODEL_HYPERPARAMETERS_FILE_PATH
from src.utils.helpers import read_yaml_file
from src.utils.helpers import load_numpy_array_data, load_object, save_object
from src.entities.config_entity import ModelTrainerConfig
from src.entities.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entities.estimator_config import MyModel



#Initiating Model Trainer Class

class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact,
                 model_training_config:ModelTrainerConfig):
       
        self.data_transformation_artifact = data_transformation_artifact
        self.model_training_config = model_training_config
        self.model_hyperparameters = read_yaml_file(MODEL_HYPERPARAMETERS_FILE_PATH)


    #For Model & Report

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:

        try:
            logging.info("Training XGBClassifier with specified parameters")

            #Splitting the Data
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            model = XGBClassifier(
            n_estimators=self.model_hyperparameters["hyperparameters"]["n_estimators"],
            max_depth=self.model_hyperparameters["hyperparameters"]["max_depth"],
            learning_rate=self.model_hyperparameters["hyperparameters"]["learning_rate"],
            subsample=self.model_hyperparameters["hyperparameters"]["subsample"])
                            

            # Fit the model
            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(accuracy=accuracy,f1_score=f1, precision_score=precision, recall_score=recall)
            return model, metric_artifact            

        except Exception as e:
            raise MyException(e, sys) from e
        


    #For Initiation

    def initiate_model_trainer(self)-> ModelTrainerArtifact:

        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")

            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")

            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")

            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_hyperparameters["Expected_Model_Score"]:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_training_config.trained_model_file_path, my_model)

            logging.info("Saved final model object that includes both preprocessing and the trained model")


            #Saving Model Parameters used to a JSON file
            model_parameters = {
                "model_name": trained_model.__class__.__name__,  
                "parameters": self.model_hyperparameters["hyperparameters"]
                }
            
            parameters_dir = os.path.dirname(self.model_training_config.trained_model_parameters_path)
            os.makedirs(parameters_dir,exist_ok=True)

            with open(self.model_training_config.trained_model_parameters_path, "w") as parameters_file:
                json.dump(model_parameters, parameters_file, indent=4)

            logging.info("Model parameters file created and saved to JSON file.")                


            # Saving model metrics to a JSON file
            model_metrics = {
                "accuracy": metric_artifact.accuracy,
                "f1_score": metric_artifact.f1_score,
                "precision_score": metric_artifact.precision_score,
                "recall_score": metric_artifact.recall_score
                }
            
            metrics_dir = os.path.dirname(self.model_training_config.trained_model_metrics_path)
            os.makedirs(metrics_dir,exist_ok=True)
                                                 

            with open(self.model_training_config.trained_model_metrics_path, "w") as metrics_file:
                json.dump(model_metrics, metrics_file, indent=4)

            logging.info("Model metrics report created and saved to JSON file.")


            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_training_config.trained_model_file_path,
                trained_model_parameters_path=self.model_training_config.trained_model_parameters_path,
                trained_model_metrics_path=self.model_training_config.trained_model_metrics_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e            