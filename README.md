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
