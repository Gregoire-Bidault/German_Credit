# German Credit — Kaggle Challenge
MSc&T — Data Science for Business (Year 1)

## Project overview
Short exploration and modelling project for the German Credit dataset, completed as part of the MSc&T Data Science for Business course. Goal: predict credit risk (good / bad) and produce a submission for a Kaggle competition-style evaluation while following a reproducible ML workflow.

## Dataset
- Source: German Credit dataset (public, often available via UCI / Kaggle).
- Problem: binary classification (creditworthy vs non-creditworthy).
- Note: raw data is not included in this repository. Download from the original source and place in the `data/` folder.

## Objectives
- Explore the dataset and understand feature distributions and class balance.
- Engineer features and preprocess categorical/continuous variables.
- Train and compare several models (logistic regression, tree-based models, ensemble methods).
- Evaluate with cross-validation and holdout; produce a Kaggle-style submission.
- Interpret model results and feature importance.

## Repository structure
- README.md — this file

Work in progress, imported from the Kaggle Challenge : 
- data/ — (not tracked) place raw CSV files here
- notebooks/
    - 01_exploration.ipynb — EDA and initial data cleaning
    - 02_feature_engineering.ipynb — preprocessing and features
    - 03_modeling.ipynb — training, CV, and evaluation
    - 04_interpretation.ipynb — feature importance and explanations
- src/
    - utils.py — loading and preprocessing helpers
    - custom_metric.py - a custom-business related metric
    - modeling.py — training and evaluation functions
- submissions/ — saved submission CSV(s)
- requirements.txt — Python package requirements

## Modelling summary
- Pipeline used: standard preprocessing (imputation, encoding, scaling) → feature selection → model training.
- Models tried: logistic regression, random forest, XGBoost/LightGBM, stacking ensemble.
- Best performing approach: tree-based ensemble with careful CV and calibration.
- Evaluation metrics: ROC-AUC and precision/recall with focus on minimizing false negatives for business impact.

## Key findings
- Several categorical features carry strong predictive signal after appropriate encoding.
- Class imbalance requires stratified CV and, in some cases, resampling or class weighting.
- Feature importance and model explainability (SHAP / permutation importance) were used to validate business-relevant predictors.