categorical_features=["ProductCD",
                      "card1", 
                      "card2", 
                      "card3", 
                      "card4", 
                      "card5", 
                      "card6",
                      "addr1", 
                      "addr2"]
target_column = 'isFraud'
confidence_interval = 0.05
no_of_trees_in_forest = 1
hyper_parameters = []
train_size = 0.7
test_size = 0.3
stratified_k_fold = "stratified-k-fold"
train_test = "train_test_split"
k_fold = "k_fold"
max_depth=25