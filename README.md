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
| `build_decision_tree.py` | file desc |
| `build_random_forest.py` | file desc |
| `build_decision_tree.py` | file desc |
| `build_random_forest.py` | file desc |
| `build_decision_tree.py` | file desc |
| `build_random_forest.py` | file desc |
| `build_decision_tree.py` | file desc |
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
