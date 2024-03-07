from utils.consts import categorical_features
from tree.Leaf import Leaf
from tree.Tree import Tree
import numpy as np
import statistics as st
import pandas as pd
from utils.model_file_utils import *
from training.build_decision_tree import handle_missing_values

def tree_classify(df: pd.DataFrame, tree: Tree):
    """ Returns the decision tree classification of the given instance

        Parameters:
            df: a single row Pandas dataframe
            tree: Either a Leaf or Node object representing the decision
                tree used to make the classification
    """

    # case 1: tree is a leaf
    if type(tree) is Leaf:
        return tree.target

    # case 2: tree is a node
    if tree.feature in categorical_features:
        for branch in tree.branches:
            if df[tree.feature] == branch.feature_value:
                return tree_classify(df, branch.tree)
        
    else:
        split_cutoff_value = tree.branches[0].feature_value.replace('<', '')
        feature_value = df[tree.feature]

        if feature_value < float(split_cutoff_value):
            return tree_classify(df, tree.branches[0].tree)
        else:
            return tree_classify(df, tree.branches[1].tree)
    
    
def forest_classify(df, forest):
    """ Classifies the given 1 row dataframe using the provided Forest model
    """
    
    classification_results = []
    for tree in forest.trees:
        classification_result = tree_classify(df, tree)
        classification_results.append(classification_result)

    return st.mode(np.array(classification_results))


def generate_predictions_file(model_file: str, output_file='output.csv'):
    """ Generates a CSV file of class predictions using a Forest model
    """

    # read forest model from file
    forest_model = read_forest_model(model_file)

    # read testing data and hanlde missing values
    testing_df = pd.read_csv('data/test.csv')
    testing_df = handle_missing_values(testing_df)

    # generate predictions, concat with transaction ids, and output to csv
    predictions = testing_df.apply(forest_classify, args=(forest_model), axis=1)
    predictions_with_ids = pd.concat([testing_df.get('TransactionID'), predictions], axis=1).reset_index(drop=True)
    predictions.to_csv(output_file, index=False)

# use this main function to generate CSV file by calling this file, just replace forest_model_name
#if __name__ == '__main__':
#    generate_predictions_file('models/forest_model_name', output_file='output.csv')
