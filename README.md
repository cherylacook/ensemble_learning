# Ensemble Learning with Random Patches

This project was *completed as part of AIML232* at Te Herenga Waka — Victoria University of Wellington.

## Objective
Implement a custom ensemble algorithm based on Random Patches (random subspaces + bagging) using decision trees as base learners. Include majority voting and a weighted voting scheme that leverages out-of-bag (OOB) accuracy to improve predictions.

## Dataset
- `electricity2.csv` - Contains the input features and target labels.

## Structure
- `random_patches_ensemble.ipynb` – Full implementation, training, evaluation, and printed results.
- `electricity2.csv` – The dataset used for experiments.
- `requirements.txt` – Python dependencies.

## Methods
- *Random Patches Ensemble*: Trains multiple decision trees on bootstrapped instances and random feature subsets (subspaces).  
- *Voting Schemes*:
   - Majority vote (default)
   - Weighted vote using each learner’s OOB accuracy: trees with higher OOB performance contribute more to final predictions.
- *Evaluation*: Computes accuracy on a held-out test set for each classifier.

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

The custom Random Patches ensemble with weighted voting achieves near-RandomForest accuracy, highlighting the utility of ensemble learners and OOB-based weighting.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
# Open `random_patches_ensemble.ipynb` and run all cells
