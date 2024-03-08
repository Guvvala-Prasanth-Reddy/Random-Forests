# Random-Forests

## Description

UNM CS 529 Project 1: Development of random forest classifier from scratch.

## Instructions for Use

### Instructions to install the dependencies
```bash
- python -m venv YOURVENV
- YOURENV/Scripts/activate
- pip install requirements.txt
```

### Train Random Forest

`python -m training.build_random_forest path/to/train.csv`

### Generate predictions using model

`python -m classification.classify path/to/model path/to/test.csv`

### Code Manifest
| File Name | Description |
| --- | --- |
| `training/build_decision_tree.py` | This file contains the implementations that build the decision tree and also the driver to call the build decision tree |
| `training/build_random_forest.py` | This file contains the implementations that build the random forest and also the driver to call the build random forest  |
| `Tree/Branch.py` | This file has a single Branch class declared that represents the branches between tree nodes  |
| `Tree/Forest.py` | This file has the declaration of the Forest class that represents a collection of the decision trees   |
| `Tree/Leaf.py` | This file has the declaration of the Leaf class we use to model the leaf nodes in the decision tree.   |
| `Tree/Node.py` | This file has the declaration of the Node class we use to define a Node in a tree other than Leaf nodes . |
| `Tree/Tree.py` | This file contains the definition of the Tree class which is the parent of the Branch, Node, and Leaf  classes. |
| `classification/classify.py` | Contains functions for classifying test instances using a trained model and for creating kaggle CSV file. |
| `validation/validate.py` | Contains functions for generating balanced accuracy and error for tree and forest models using test data split. |
| `utils/consts.py` | Contains global constants (e.g. hyperparameters) used throughout the project. |
| `utils/dataframeutils.py` | Contains functions for performing splits and sampling on Pandas dataframes. |
| `util/impurity.py` | Contains functions for calculating information gain and Chi Square test value. |
| `util/model_file_utils.py` | Contains functions for reading models from file and for performing tree structure analysis. |


## Developer Contributions

Prasanth Guvvala
- Implemented handling of missing value functions.
- Implemented information gain criteria methods.
- Implemented custom tree data objects.
- Implemented methods to sample and split data into testing and training sets.

Thomas Fisher
- Implemented function to build decision tree.
- Implemented validation functions for trees and forests.
- Implemented Chi Square termination test.
- Implemented functions to analyze tree structure.

## kaggle Submission

Leaderboard position 7 achieved with accuracy 0.75530 on March 7th.
