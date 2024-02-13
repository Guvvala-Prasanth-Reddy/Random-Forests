from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
import pandas as pd


def train_test_data_split(df : pd.DataFrame , mode : str ,  no_of_random_parameters : int , column_list : list[str]  , number_of_splits : int )  -> pd.DataFrame : 
    if( mode == "K-fold"):
        return df[column_list] 
    if( mode == "stratified-K-fold"):
        return
    if( mode == "train_test_data_split"):

        return 