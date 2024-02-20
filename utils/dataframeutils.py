from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils.consts import *
import random
import pandas as pd
import numpy as np


def train_test_data_split(df : pd.DataFrame , mode : str = train_test , column_list : list[str] = []  , number_of_folds : int = 10 ) -> pd.DataFrame :
    """ Returns a decision tree object built upon the provided data.

        Parameters:
            df: a Pandas dataframe
            mode : indicates the splitting mode to use
            column_list : return dataframe only with these columns
            number_of_splits : it is the no of folds that the dataframe is splitted into 
                with information gain. Options are the following:
                    'entropy': uses entropy
                    'gini': uses Gini index
                    'misclassification': uses misclassification error

        Returns:
            A data frame that is split on basis of mode

    """
    if( column_list != []):
        df = df[column_list] 
    if( mode == k_fold):
        return KFold(n_splits= number_of_folds ).split( X = df.drop(columns=[target_column]) , y = df[target_column] )
        #Done : return number_of_splits:k  with given column_list
    
    if( mode == stratified_k_fold):

        #Done: return stratified K splits with given columns 
        return StratifiedKFold( n_splits=number_of_folds , shuffle=False).split( X = df.drop(columns=[target_column]) , y = df[target_column] )
    
    if( mode == train_test):
        #Done : return train_test_data in given split percentages 

        return train_test_split(  df.drop(columns= [target_column])  ,  df.get(target_column) , test_size = test_size , train_size =  train_size , stratify= df.get(target_column))
    

def random_sample_columns(df : pd.DataFrame , sample_size : int )-> list[str] :
    """
    Parameters :
        df : dataframe 
        sample_size : int  

    Returns:
        a random sample of columns of size sample_size
    """
    return( random.sample(population = df.columns , k = sample_size))


def random_sample_rows(df : pd.DataFrame , sample_size : int ):

    """
    Parameters :
        df : dataframe
        sample_size : int 
    Returns :
        a random sample of rows of size sample_size : int
    """
    return df.sample( n = sample_size )


#some line to test the code....
# if __name__ == "__main__":
#     df = pd.read_csv("data/train.csv")
#     X_train , X_test , y_train, y_test = train_test_data_split(df , mode = train_test )
#     print( X_train , y_train)
def split_data(df : pd.DataFrame , column_sample_size , row_sample_size, replace):

    columns = df.columns
    columns = columns.drop( labels=["TransactionID", "isFraud"] )
    columns = list(columns)
    columns = random.sample(columns, k = int(column_sample_size*len(columns)))
    columns.append("TransactionID")
    columns.append( "isFraud" )
    df = df[columns]

    # do row bagging here
    df = df.sample(frac=row_sample_size, replace=replace)


    X_train , X_test , y_train, y_test = train_test_data_split(df , mode = train_test )
    return (pd.concat([X_train , y_train] , axis = 1).reset_index(drop=True) , pd.concat([X_test ,y_test] , axis=1).reset_index(drop=True))

