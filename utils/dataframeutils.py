import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils.consts import *


MISSING_VALUE_TERMS = ['notFound', float('NaN'), 'NaN']

def handle_missing_values_test(df: pd.DataFrame) -> pd.DataFrame:
    """ Handles missing values in the dataset where the dataset does not contain a 
        target column (testing data)

        Parameters:
            df: a pandas DataFrame

        Returns:
            a copy of the provided DataFrame, but with missing values handled in a way
                to maintain trainin accuracy
    """
    
    # convert all missing value types (i.e. 'NotFound' strings) to float NaNs
    df.replace('NotFound', float('nan'), inplace=True)

    # create a dictionary mapping feature columns to their mode/mean; then replace missing
    # values based on the most common value in their column
    feature_missing_value_replacement_dict = dict()
    for feature in df.columns:
        if feature in categorical_features:
            feature_missing_value_replacement_dict[feature] = df.get(feature).mode()[0]
        else:
            feature_missing_value_replacement_dict[feature] = df.get(feature).mean()

    df.fillna(value=feature_missing_value_replacement_dict, inplace=True)

    return df


def handle_missing_values_train(df: pd.DataFrame) -> pd.DataFrame:
    """ Handles missing values in the dataset where the dataset does contain a 
        target column (training data)

    Parameters:
        df: a Pandas dataframe

    Returns:
        The provided dataframe without missing values
    """

    # remove instances for which the target is not known
    for i in range(len(MISSING_VALUE_TERMS)):
        df = df.loc[df[target_column] != MISSING_VALUE_TERMS[i]]

    # convert all missing value types (i.e. 'NotFound' strings) to float NaNs
    df.replace('NotFound', float('nan'), inplace=True)

    # create a 2D dictionary where feature column mean/modes are calculated according
    # to target class; then assign values to missing data based on their feature column
    # and class
    feature_missing_value_replacement_dict = dict()
    for target_val in pd.unique(df[target_column]):
        feature_missing_value_replacement_dict[target_val] = dict()
        for feature in df.columns:
            if feature in categorical_features:
                feature_missing_value_replacement_dict[target_val][feature] = df.loc[df[target_column] == target_val].get(feature).mode()[0]
            else:
                feature_missing_value_replacement_dict[target_val][feature] = df.loc[df[target_column] == target_val].get(feature).mean()

    for target in pd.unique(df[target_column]):
        df1 = df.loc[df[target_column] == target].fillna(value = feature_missing_value_replacement_dict[target]).copy()
        df.loc[df1.index] = df1

    return df


def train_test_data_split(df: pd.DataFrame, mode: str=train_test, column_list: list[str]=[], number_of_folds: int=10 ) -> pd.DataFrame:
    """ Splits the provided DataFrake using one of several split methods

        Parameters:
            df: a pandas DataFrame
            mode: indicates the splitting mode to use
            column_list: if included, indicates that only these columns should be included
                in the result
            number_of_folds: the number of folds into which the DataFrame is split

        Returns:
            A DataFrame that is split on basis of mode

    """

    if(column_list != []):
        df = df[column_list] 
    if(mode == k_fold):
        return KFold(n_splits= number_of_folds).split(X = df.drop(columns=[target_column]), y = df[target_column])    
    if(mode == stratified_k_fold):
        return StratifiedKFold(n_splits=number_of_folds, shuffle=False).split(X=df.drop(columns=[target_column]),
                                                                              y=df[target_column])
    if(mode == train_test):
        return train_test_split(df.drop(columns=[target_column]), df.get(target_column), 
                                test_size=test_size, train_size=train_size, stratify=df.get(target_column))
    

def random_sample_columns(df: pd.DataFrame, sample_size: int) -> list[str]:
    """ Returns a random sampling without replacement of the provided DataFrame

    Parameters:
        df: a pandas DataFrame 
        sample_size: the number of columns to randomly sample, where
            sample_size <= len(df.columns)

    Returns:
        a random sample of column names as a list of strings of size sample_size
    """

    return(random.sample(population=df.columns, k=sample_size))


def random_sample_rows(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """ Returns a random sampling of the rows of the provided DataFrame without
        replacement

    Parameters:
        df: a pandas DataFrame
        sample_size: the number of rows to randomly sample, where
            sample_size <= len(df)

    Returns:
        a random sample of sample_size rows from df
    """

    return df.sample(n=sample_size)


def split_data(df: pd.DataFrame, column_sample_frac: float, row_sample_frac: float, replace: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Samples the provided DataFrame and splits it into test and train portions

        Parameters:
            df: a pandas DataFrame
            column_sample_frac: a float between 0.0 and 1.0 indicating what fraction of columns
                to keep when sampling the columns of the DataFrame
            row_sample_frac: a float between 0.0 and 1.0 indicating what fraction of rows to keep
                when sampling the rows of the DataFrame
            replace: a boolean value indicating whether sampling is to be performed with replacement
        
        Returns:
            a tuple of the training and testing DataFrames generated from the provided DataFrame
    """

    # sample the indicated fraction of the feature columns of the DataFrame
    columns = df.columns
    columns = columns.drop(labels=["TransactionID", "isFraud"])
    columns = list(columns)
    columns = random.sample(columns, k=int(column_sample_frac * len(columns)))
    columns.append("TransactionID")
    columns.append("isFraud")
    df = df[columns]

    # sample the rows of the DataFrame according to the provided parameters
    df = df.sample(frac=row_sample_frac, replace=replace)

    # split sampled data into test and train portions
    X_train, X_test, y_train, y_test = train_test_data_split(df, mode=train_test)
    return (pd.concat([X_train, y_train], axis=1).reset_index(drop=True), 
            pd.concat([X_test, y_test], axis=1).reset_index(drop=True))

