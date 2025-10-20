import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, precision_score
from sklearn.model_selection import cross_val_score
from skrub import TableVectorizer
from xgboost import XGBClassifier
import utils
import Custom_metric
import optuna


def explore_categorical_column(df: pd.DataFrame, col: str) -> None:
    """
    Explores a categorical or discrete column in a dataframe by printing unique values, value counts, and missing values.

    Args:
        df (pd.DataFrame): The dataframe containing the column to explore.
        col (str): The name of the column to explore.
    Returns:
        Unique values, value counts, and missing values in the specified column.
    """
    print(f"Unique values in {col}: {df[col].unique()}")
    print(f"Value counts in {col}:\n{df[col].value_counts()}")
    print(f"Missing values in {col}: {df[col].isnull().sum()}")


def explore_numerical_column(df: pd.DataFrame, col: str) -> None:
    """
    Explores a numeric column in a dataframe by printing unique values, statistics, and missing values.

    Args:
        df (pd.DataFrame): The dataframe containing the column to explore.
        col (str): The name of the column to explore.
    Returns:
        Unique values, value counts, and others in the specified column.
    """
    print(f"Unique values in {col}: {df[col].unique()}")
    print(f"Descriptive statistics for {col}:\n{df[col].describe()}")
    print(f"Missing values in {col}: {df[col].isnull().sum()}")
    print(f"Null values for {col}: {df[col].isnull().sum()}")
    print(df[col].hist())
    print(f"Duplicated values for {col}: {df[col].duplicated().sum()}")


def submission(filename: str, df: pd.DataFrame, y_pred: np.array) -> None:
    """
    Saves the dataframe as a csv file.

    Args:
        df (pd.DataFrame): The dataframe to save.
        filename (str): The name of the file to save.
    Returns:
        None
    """
    sub_file = pd.DataFrame({"Id": df.index, "Risk": y_pred})
    sub_file["Risk"] = sub_file["Risk"].apply(lambda x: "Risk" if x == 1 else "No Risk")
    sub_file = sub_file.to_csv(filename, index=False)


def create_train_test():
    """
    Creates the train and test datasets.

    Args:
        None
    Returns:
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training labels.
        test (pd.DataFrame): The test data.
    """
    train = pd.read_csv("german_credit_train.csv")
    test = pd.read_csv("german_credit_test.csv")

    train["Risk"] = train["Risk"].map({"No Risk": 0, "Risk": 1})
    X_train = train.drop(columns="Risk")
    y_train = train["Risk"]

    return X_train, y_train, test


def objective(trial : optuna.trial.Trial, X_train : pd.DataFrame, y_train : pd.Series, metric : str) -> float:
    """
    This function is used to tune the hyperparameters of the XGBoost classifier.

    Args:
        trial (optuna.trial.Trial):
        X_train (pd.DataFrame): the dataframe containing the features
        y_train (pd.Series): the target variable
        metric (str): the metric to optimize

    Returns:
        float: the recall score of the model
    """
    
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    # First we do a preprocessing step
    preprocessor = ColumnTransformer(
        transformers=[(
            "vectorizer", TableVectorizer(), X_train.columns
        )
        ]
    )

    # Then we create the model
    model = XGBClassifier(**params)

    # Last we create the pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=metric, n_jobs=-1)
    
    return -scores.mean()


def optimise_with_optuna(X_train : pd.DataFrame, y_train : pd.Series, metric : str, n_trials=10) -> Pipeline:
    """
    Create a pipeline with TableVectorizer and XGBClassifier with Optuna for hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.
        n_trials (int, optional): Number of trials for Optuna. Defaults to 10.

    Returns:
        Pipeline: The best pipeline found by Optuna.
    """
    # Initialize Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, metric), n_trials=n_trials)

    # Get the best parameters
    best_params = study.best_params
    best_params["tree_method"] = "hist"

        # First we do a preprocessing step
    preprocessor = ColumnTransformer(
        transformers=[(
            "vectorizer", TableVectorizer(), X_train.columns
        )
        ]
    )

    # Then we create the model
    model = XGBClassifier(**best_params)

    # Last we create the pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline