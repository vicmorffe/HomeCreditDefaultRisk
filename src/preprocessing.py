from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    datasets = dict(train=working_train_df, val=working_val_df, test=working_test_df)

    cat_features_train = working_train_df.select_dtypes("object")
    train_binary_features = cat_features_train.loc[:, cat_features_train.nunique() == 2]
    train_non_binary_features = cat_features_train.loc[
        :, cat_features_train.nunique() > 2
    ]

    binary_columns = train_binary_features.columns
    non_binary_columns = train_non_binary_features.columns

    ordinal_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan
    )
    ordinal_enc.fit(train_binary_features)
    one_hot_enc = OneHotEncoder(handle_unknown="ignore")
    one_hot_enc.fit(train_non_binary_features)
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()

    for name, df in datasets.items():
        binary_enc_features = ordinal_enc.transform(df[binary_columns])
        non_binary_enc_features = one_hot_enc.transform(
            df[non_binary_columns]
        ).toarray()

        df.drop(columns=binary_columns, inplace=True)
        df.drop(columns=non_binary_columns, inplace=True)
        df_array = df.to_numpy()

        datasets[name] = np.concatenate(
            (df_array, binary_enc_features, non_binary_enc_features), axis=1
        )


    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

        if name == "train":
            imputer.fit(datasets[name])
            scaler.fit(datasets[name])
        datasets[name] = imputer.transform(datasets[name])

        # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
        # Please use sklearn.preprocessing.MinMaxScaler().
        # Again, take into account that:
        #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
        #     working_test_df).
        #   - In order to prevent overfitting and avoid Data Leakage you must use only
        #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
        #     model to transform all the datasets.
    
        datasets[name] = scaler.transform(datasets[name])

    return list(datasets.values())


    
