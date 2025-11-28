# Ensemble Learning with Random Patches

This project was *completed as part of AIML232* at Te Herenga Waka — Victoria University of Wellington.

## Objective
Implement an ensemble algorithm based on Random Patches (random subspaces + bagging) using decision trees as base learners. Include majority voting and a weighted voting scheme based on out-of-bag (OOB) accuracy.

## Dataset
- `electricity2.csv` - Contains the input features and target labels.

## Structure
- `ensemble_learning.ipynb` – Contains the full implementation, training, and evaluation.
- `electricity2.csv` – The dataset used for experiments.
- `requirements.txt` – Python dependencies.

## Methods
- *Random Patches Ensemble*: Trains multiple decision trees on bootstrapped instances and random feature subsets.  
- *Voting Schemes*: Default majority vote; optional weighted vote using OOB accuracy.  
- *Evaluation*: Computes accuracy on a test set and prints OOB performance per learner.

## Results
Accuracy on the Electricity dataset:

| Classifier                          | Accuracy |
|-------------------------------------|----------|
| DecisionTree                         | 0.752    |
| Bagging                              | 0.805    |
| RandomForest                          | 0.811    |
| AdaBoost                              | 0.762    |
| XGBoost                               | 0.804    |
| RandomPatches(11, 50) (Weighted Vote)| 0.808    |

The custom Random Patches ensemble with weighted voting performs competitively with standard ensemble methods.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
# Open `random_patches_ensemble.ipynb` and run all cells
