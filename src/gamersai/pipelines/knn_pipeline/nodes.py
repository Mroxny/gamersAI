"""
This is a boilerplate pipeline 'knn_pipeline'
generated using Kedro 0.19.9
"""
import logging
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import wandb
import wandb.sklearn
import os

def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> KNeighborsRegressor:
    """Trains a K-Nearest Neighbors Regressor model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.

    Returns:
        Trained K-Nearest Neighbors model.
    """
    #model = KNeighborsRegressor(n_neighbors=8)
    model = KNeighborsRegressor(n_neighbors=parameters["n_neighbors"])
    model.fit(X_train, y_train)
    
    #wandb.sklearn.plot_regressor(model=model, X_train=X_train, X_test=X_test,y_train=y_train,y_test=y_test)
    
    
    return model

def evaluate_knn_model(
    regressor: KNeighborsRegressor, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the KNN model.

    Args:
        regressor: Trained K-Nearest Neighbors model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gamersAI",
        name = "KNN",
        group=os.environ["WANDB_RUN_GROUP"],
        # track hyperparameters and run metadata
        config=regressor.get_params()
    )
    wandb.sklearn.plot_learning_curve(model=regressor, X=X_train,y=y_train)
    wandb.sklearn.plot_summary_metrics(regressor, X_train, y_train, X_test, y_test)
    run = wandb.run
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("KNN model has a coefficient R^2 of %.3f on test data.", score)
    to_log = {
            "mse" : mse,
            "mae" : mae,
            "R2 score":score
              }
    run.log(to_log)
    run.finish()
