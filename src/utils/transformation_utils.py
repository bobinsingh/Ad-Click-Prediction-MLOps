# import pandas as pd
# import sys
# from src.logging import logging
# from src.exceptions import MyException
# from sklearn.impute import KNNImputer


# # Encodes Categorical Features

# def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Encodes categorical features in the DataFrame using one-hot encoding with drop_first=True.
    
#     Args:
#         df (pd.DataFrame): Input DataFrame containing categorical features.
    
#     Returns:
#         pd.DataFrame: DataFrame with one-hot encoded categorical features.
#     """
#     try:
#         # List of categorical columns to encode
#         categorical_columns = ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']
        
#         logging.info(f"Starting one-hot encoding for columns: {categorical_columns}")

#         # Check if all specified columns are present in the DataFrame
#         missing_columns = [col for col in categorical_columns if col not in df.columns]
#         if missing_columns:
#             logging.warning(f"Missing categorical columns: {missing_columns}")
#             raise ValueError(f"The following categorical columns are missing from the DataFrame: {missing_columns}")

#         # Perform one-hot encoding with drop_first=True
#         df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
#         logging.info("One-hot encoding completed successfully.")

#         return df_encoded

#     except Exception as e:
#         logging.error("Error occurred during one-hot encoding.")
#         raise MyException(e, sys)



# # Handles Missing Values and Fill them

# def fill_na_and_knn_impute(data: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
#     """
#     Fills NaN values in categorical columns with 'Unknown', applies KNN imputation for numerical columns, 
#     and restores original categories for categorical columns.
    
#     Args:
#         data (pd.DataFrame): Input DataFrame.
#         n_neighbors (int): Number of neighbors for KNN imputation.
    
#     Returns:
#         pd.DataFrame: DataFrame with filled missing values and KNN-imputed numerical data.
#     """
#     try:
#         # Fill NaN values in categorical columns with 'Unknown'
#         categorical_columns = data.select_dtypes(include='object').columns
#         for col in categorical_columns:
#             data[col] = data[col].fillna('Unknown')
#             logging.info(f"Filled NaN in column '{col}' with 'Unknown'.")

#         logging.info("NaN values in categorical columns filled with 'Unknown'. Proceeding with KNN imputation.")

#         # Encode categorical columns for KNN imputation
#         data_encoded = data.copy()
#         category_mappings = {}

#         for col in categorical_columns:
#             data_encoded[col] = data_encoded[col].astype('category').cat.codes
#             category_mappings[col] = dict(enumerate(data[col].astype('category').cat.categories))
#             logging.debug(f"Encoded column '{col}' for KNN imputation.")

#         # Apply KNN imputer to the dataset
#         knn_imputer = KNNImputer(n_neighbors=n_neighbors)
#         data_imputed = pd.DataFrame(knn_imputer.fit_transform(data_encoded), columns=data_encoded.columns)

#         logging.info(f"KNN imputation completed with n_neighbors={n_neighbors}.")

#         # Decode categorical columns back to original categories
#         for col in categorical_columns:
#             data_imputed[col] = (
#                 data_imputed[col]
#                 .round()
#                 .astype(int)
#                 .map(category_mappings[col])
#             )
#             logging.debug(f"Decoded column '{col}' back to original categories.")

#         logging.info("All categorical columns decoded back to original categories.")
#         return data_imputed

#     except Exception as e:
#         logging.error("Error occurred during filling NaN and KNN imputation.")
#         raise MyException(e, sys)


# #For Dropping Unnecessary Columns

# def drop_columns(df, schema_config):
#     """Drop the specified columns if they exist."""
#     logging.info("Dropping columns: %s", schema_config['drop_columns'])
    
#     drop_cols = schema_config['drop_columns']
    
#     # Loop through each column in the 'drop_columns' list
#     for col in drop_cols:
#         if col in df.columns:
#             logging.info(f"Dropping column: {col}")
#             df = df.drop(col, axis=1)
#         else:
#             logging.warning(f"Column '{col}' not found in DataFrame")
    
#     return df


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
import pandas as pd
import sys
from src.logging import logging
from src.exceptions import MyException

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, schema_config):
        self.schema_config = schema_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            drop_cols = self.schema_config['drop_columns']
            for col in drop_cols:
                if col in X.columns:
                    logging.info(f"Dropping column: {col}")
                    X = X.drop(col, axis=1)
                else:
                    logging.warning(f"Column '{col}' not found in DataFrame")
            return X
        except Exception as e:
            logging.error("Error occurred during column dropping.")
            raise MyException(e, sys)
        

class EncodeCategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, schema_config):
        """
        Initializes the transformer with the schema configuration.

        Args:
            schema_config (dict): Dictionary containing configuration for the schema, including categorical columns.
        """
        self.schema_config = schema_config

    def fit(self, X, y=None):
        """
        Extracts the categorical columns from the schema configuration and checks their presence in the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series or None): Target variable (not used here).

        Returns:
            self: Fitted transformer.
        """
        try:
            # Extract categorical columns from schema_config
            self.categorical_columns = self.schema_config.get("categorical_columns", [])
            
            # Validate if all specified categorical columns exist in the DataFrame
            missing_columns = [col for col in self.categorical_columns if col not in X.columns]
            if missing_columns:
                raise ValueError(f"The following categorical columns are missing: {missing_columns}")
            
            logging.info(f"Categorical columns identified: {self.categorical_columns}")
            return self
        except Exception as e:
            logging.error("Error occurred during the fit of EncodeCategoricalFeatures.")
            raise MyException(e, sys)

    def transform(self, X):
        """
        Performs one-hot encoding on the specified categorical columns.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded categorical columns.
        """
        try:
            logging.info(f"Starting one-hot encoding for columns: {self.categorical_columns}")
            return pd.get_dummies(X, columns=self.categorical_columns, drop_first=True)
        except Exception as e:
            logging.error("Error occurred during one-hot encoding.")
            raise MyException(e, sys)



class FillNaAndKNNImpute(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self.categorical_columns = X.select_dtypes(include='object').columns
        self.category_mappings = {}

        X_encoded = X.copy()
        for col in self.categorical_columns:
            X_encoded[col] = X_encoded[col].astype('category').cat.codes
            self.category_mappings[col] = dict(enumerate(X[col].astype('category').cat.categories))

        self.knn_imputer.fit(X_encoded)
        return self

    def transform(self, X):
        try:
            X_encoded = X.copy()
            for col in self.categorical_columns:
                X_encoded[col] = X_encoded[col].astype('category').cat.codes

            X_imputed = pd.DataFrame(self.knn_imputer.transform(X_encoded), columns=X.columns)

            for col in self.categorical_columns:
                X_imputed[col] = (
                    X_imputed[col]
                    .round()
                    .astype(int)
                    .map(self.category_mappings[col])
                )
            return X_imputed
        except Exception as e:
            logging.error("Error occurred during KNN imputation.")
            raise MyException(e, sys)
