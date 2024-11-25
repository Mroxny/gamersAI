"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.9
"""
import logging
from typing import Tuple

import pandas as pd
import wandb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import wandb.sklearn
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold


def split_data(data: pd.DataFrame, parameters: dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters_data_science.yml.
    Returns:
        Split data.
    """

    X = data.drop(parameters["features"], axis='columns')
    y = data['Estimated owners (avg)']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"], random_state=parameters["random_state"])

    return X_train, X_test, y_train, y_test


def train_model(int_5: int,X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> RandomForestRegressor:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    #wandb.sklearn.plot_regressor(model=model, X_train=X_train, X_test=X_test,y_train=y_train,y_test=y_test)
    
   
    return model


def evaluate_model(
    regressor: RandomForestRegressor, X_train:pd.DataFrame,y_train:pd.Series, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gamersAI",
        name = "RF",
        group=os.environ["WANDB_RUN_GROUP"],
        # track hyperparameters and run metadata
        config=regressor.get_params()
    )
    wandb.sklearn.plot_learning_curve(model=regressor, X=X_train,y=y_train)
    #wandb.sklearn.plot_summary_metrics(regressor, X_train, y_train, X_test, y_test)
    wandb.sklearn.plot_residuals(regressor,X_train,y_train)
    df_seperated_X_numerical = X_train.select_dtypes(include=['number','boolean'])
    df_seperated_X_numerical = X_train.replace({False: 0,True: 1})
    wandb.sklearn.plot_outlier_candidates(regressor,df_seperated_X_numerical,y_train)
    run = wandb.run
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("RF model has a coefficient R^2 of %.3f on test data.", score)
    to_log = {
            "mse" : mse,
            "mae" : mae,
            "R2 score":score
              }
    run.log(to_log)
    run.finish()
    return 1

def cross_validate_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    Performs cross-validation for the RandomForestRegressor and logs the R² scores to WandB.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.
        parameters: Hyperparameters for the RandomForestRegressor.
    """
    # Define the Random Forest model
    model = RandomForestRegressor(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters.get("max_depth"),
        random_state=parameters["random_state"]
    )

    # Initialize WandB
    run = wandb.init(
        project="gamersAI",
        name="RF_CrossValidation",
        group=os.environ.get("WANDB_RUN_GROUP"),
        config=model.get_params()
    )

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=101)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    # Log R² scores chart
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b')
    plt.title("RandomForest Cross-Validation R² Scores")
    plt.xlabel("Fold")
    plt.ylabel("R² Score")
    plt.grid()
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.ylim(-0.1, 1.1) 

    # Log the plot to WandB
    wandb.log({"R² Scores Plot": wandb.Image(plt)})
    plt.close()

    # Log the results to the console
    logger = logging.getLogger(__name__)
    logger.info("RandomForest Cross-validation R² scores: %s", cv_scores)
    logger.info("RandomForest Mean R² score: %.3f", cv_scores.mean())

    run.finish()

   