# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.impute import KNNImputer
# import pandas as pd
# import sys
# from src.logging import logging
# from src.exceptions import MyException

# class DropColumns(BaseEstimator, TransformerMixin):
#     def __init__(self, schema_config):
#         self.schema_config = schema_config

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         try:
#             drop_cols = self.schema_config['drop_columns']
#             for col in drop_cols:
#                 if col in X.columns:
#                     logging.info(f"Dropping column: {col}")
#                     X = X.drop(col, axis=1)
#                 else:
#                     logging.warning(f"Column '{col}' not found in DataFrame")
#             return X
#         except Exception as e:
#             logging.error("Error occurred during column dropping.")
#             raise MyException(e, sys)
        

# class EncodeCategoricalFeatures(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         """
#         Initializes the transformer without requiring predefined schema configurations.
#         """
#         pass

#     def fit(self, X, y=None):
#         """
#         Identifies categorical columns from the input DataFrame.

#         Args:
#             X (pd.DataFrame): Input DataFrame.
#             y (pd.Series or None): Target variable (not used here).

#         Returns:
#             self: Fitted transformer.
#         """
#         try:
#             # Automatically identify categorical columns
#             self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
#             logging.info(f"Categorical columns identified: {self.categorical_columns}")
#             return self
#         except Exception as e:
#             logging.error("Error occurred during the fit of EncodeCategoricalFeatures.")
#             raise MyException(e, sys)

#     def transform(self, X):
#         """
#         Performs one-hot encoding on the identified categorical columns.

#         Args:
#             X (pd.DataFrame): Input DataFrame.

#         Returns:
#             pd.DataFrame: DataFrame with one-hot encoded categorical columns.
#         """
#         try:
#             logging.info(f"Starting one-hot encoding for columns: {self.categorical_columns}")
            
#             # Logging column names before encoding
#             original_columns = X.columns.tolist()
#             logging.info(f"Original columns: {original_columns}")
            
#             # Perform one-hot encoding
#             X_encoded = pd.get_dummies(X, columns=self.categorical_columns, drop_first=True)
            
#             # Logging column names after encoding
#             encoded_columns = X_encoded.columns.tolist()
#             logging.info(f"Columns after encoding: {encoded_columns}")
            
#             return X_encoded
#         except Exception as e:
#             logging.error("Error occurred during one-hot encoding.")
#             raise MyException(e, sys)




# class FillNaAndKNNImpute(BaseEstimator, TransformerMixin):
#     def __init__(self, n_neighbors=5):
#         self.n_neighbors = n_neighbors
#         self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

#     def fit(self, X, y=None):
#         self.categorical_columns = X.select_dtypes(include='object').columns
#         self.category_mappings = {}

#         X_encoded = X.copy()
#         for col in self.categorical_columns:
#             X_encoded[col] = X_encoded[col].astype('category').cat.codes
#             self.category_mappings[col] = dict(enumerate(X[col].astype('category').cat.categories))

#         self.knn_imputer.fit(X_encoded)
#         return self

#     def transform(self, X):
#         try:
#             X_encoded = X.copy()
#             for col in self.categorical_columns:
#                 X_encoded[col] = X_encoded[col].astype('category').cat.codes

#             X_imputed = pd.DataFrame(self.knn_imputer.transform(X_encoded), columns=X.columns)

#             for col in self.categorical_columns:
#                 X_imputed[col] = (
#                     X_imputed[col]
#                     .round()
#                     .astype(int)
#                     .map(self.category_mappings[col])
#                 )
#             return X_imputed
#         except Exception as e:
#             logging.error("Error occurred during KNN imputation.")
#             raise MyException(e, sys)


from sklearn.impute import KNNImputer
import pandas as pd
import sys
from src.logging import logging
from src.exceptions import MyException

def drop_columns(df, schema_config):
    """
    Drops specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        schema_config (dict): Dictionary containing the columns to drop under 'drop_columns'.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    try:
        drop_cols = schema_config['drop_columns']
        for col in drop_cols:
            if col in df.columns:
                logging.info(f"Dropping column: {col}")
                df = df.drop(col, axis=1)
            else:
                logging.warning(f"Column '{col}' not found in DataFrame")
        return df
    except Exception as e:
        logging.error("Error occurred during column dropping.")
        raise MyException(e, sys)


def encode_categorical_features(df):
    """
    Performs one-hot encoding on categorical columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical columns.
    """
    try:
        # Automatically identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        logging.info(f"Categorical columns identified: {categorical_columns}")

        # Logging column names before encoding
        original_columns = df.columns.tolist()
        logging.info(f"Original columns: {original_columns}")

        # Perform one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        # Logging column names after encoding
        encoded_columns = df_encoded.columns.tolist()
        logging.info(f"Columns after encoding: {encoded_columns}")

        return df_encoded
    except Exception as e:
        logging.error("Error occurred during one-hot encoding.")
        raise MyException(e, sys)


def fill_na_and_knn_impute(df, n_neighbors=5):
    """
    Fills missing values using KNN imputation.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.
        n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    try:
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)

        # Identify categorical columns and encode them
        categorical_columns = df.select_dtypes(include='object').columns
        category_mappings = {}

        df_encoded = df.copy()
        for col in categorical_columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
            category_mappings[col] = dict(enumerate(df[col].astype('category').cat.categories))

        # Fit the KNN imputer
        logging.info(f"Fitting KNN imputer with n_neighbors={n_neighbors}")
        knn_imputer.fit(df_encoded)

        # Transform the data
        df_imputed = pd.DataFrame(knn_imputer.transform(df_encoded), columns=df.columns)

        # Map categorical columns back to original categories
        for col in categorical_columns:
            df_imputed[col] = (
                df_imputed[col]
                .round()
                .astype(int)
                .map(category_mappings[col])
            )

        return df_imputed
    except Exception as e:
        logging.error("Error occurred during KNN imputation.")
        raise MyException(e, sys)
