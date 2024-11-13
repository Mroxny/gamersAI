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


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> RandomForestRegressor:
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
    
   