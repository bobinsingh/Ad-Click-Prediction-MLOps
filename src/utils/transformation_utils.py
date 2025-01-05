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
        # Logging column names before encoding
        original_columns = df.columns.tolist()
        logging.info(f"Original columns: {original_columns}")

        # Perform one-hot encoding
        df_encoded = pd.get_dummies(df,drop_first=True)

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
