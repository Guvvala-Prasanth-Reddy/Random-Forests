import pandas as pd
from scipy.stats import chi2

def build_tree(df, split_metric='entropy'):
    """ Returns a decision tree object built upon the provided data.

        Parameters:
            df: a Pandas dataframe
            split_metric (str): indicates which split metric to use
                with information gain. Options are the following:
                    'entropy': uses entropy
                    'gini': uses Gini index
                    'misclassification': uses misclassification error

        Returns:
            A decision tree object

    """

    # check whether all targets are the same in the provided dataset
    target_col_as_list = df['isFraud'].tolist()
    if len(set(target_col_as_list)) == 1:
        print('All the targets are the same')
        return Leaf(target_col_as_list[0])

    # assign a score to each possible data split
    # QUESTION: is IG also included in the function calls below?
    feature_scores = dict()
    for feature in df.columns:
        if feature != 'isFraud':
            if split_metric == 'entropy':
                feature_scores[feature] = get_entropy_score(df, feature)
            elif split_metric == 'gini':
                feature_scores[feature] = get_gini_score(df, feature)
            else:
                feature_scores[feature] = get_misclassification_score(df, feature)

    # determine which feature has the max information gain
    max_score_feature = ''
    max_feature_score = float('-inf')
    for feature in feature_scores.keys:
        if feature_scores[feature] > max_feature_score:
            max_feature_score = feature_scores[feature]
            max_score_feature = feature

    # check if split is recommended by chi squared test
    split_chi_squared_metric = get_chi_squared_metric(df, max_score_feature)
    degrees_of_freedom = (len(pd.unique(df[max_score_feature])) - 1) * (len(pd.unique(df['isFraud'])) - 1)
    chi_squared_table_value = chi2.ppf(0.05, degrees_of_freedom)
    if chi_squared_table_value > split_chi_squared_metric:
        return Leaf(target_col_as_list[0])
    
    # create branches from the node for all attributes of the selected feature
    node = Node(max_score_feature)
    for feature_value in pd.unique(df[max_score_feature]):
        branch = Branch(feature_value,
                        build_tree(df[max_score_feature] == feature_value, split_metric))
        node.add_branch(branch)
    
    return node









if __name__ == "__main__":
    """
    Main method for testing
    """

    df = pd.read_csv("../data/train.csv")
    build_tree(df)
