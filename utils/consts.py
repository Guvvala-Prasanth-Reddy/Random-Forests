# path to training data file
training_data_path = 'data/train.csv'

# path to testing data file
testing_data_path = 'data/test.csv'

# represents the set of categorical features from our dataset
categorical_features=["ProductCD",
                      "card1", 
                      "card2", 
                      "card3", 
                      "card4", 
                      "card5", 
                      "card6",
                      "addr1", 
                      "addr2"]

# represents the class column from our dataset
target_column = 'isFraud'

# the alpha value used in our chi square test
chi_square_alpha = 0.05

# number of trees used in forest
no_of_trees_in_forest = 1

# represent the train/test split proportion for our algorithm
train_size = 0.7
test_size = 1 - train_size

# data split parameters used for calls to sklearn data split functions
stratified_k_fold = "stratified-k-fold"
train_test = "train_test_split"
k_fold = "k_fold"

# the max depth allowed for all trees in forest
max_depth=25