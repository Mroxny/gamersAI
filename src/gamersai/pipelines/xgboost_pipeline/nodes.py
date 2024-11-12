"""
This is a boilerplate pipeline 'xgboost_pipeline'
generated using Kedro 0.19.9
"""
import logging
import pandas as pd
import wandb
import wandb.sklearn
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from wandb.integration.xgboost import WandbCallback

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> XGBRegressor:
    """Trains the XGBoost regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.

    Returns:
        Trained XGBoost model.
    """
    #model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
    model = XGBRegressor(objective=parameters["objective"], n_estimators=parameters["n_estimators"], learning_rate=parameters["learning_rate"], max_depth=parameters["max_depth"])
    
    model.fit(X_train, y_train)
    
    
    return model

def evaluate_xgboost_model(
    regressor: XGBRegressor, X_train:pd.DataFrame,y_train:pd.Series, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the XGBoost model.

    Args:
        regressor: Trained XGBoost model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gamersAI",
        name = "XGBoost",
        # track hyperparameters and run metadata
        config=regressor.get_params()
    )
    wandb.sklearn.plot_learning_curve(model=regressor, X=X_train,y=y_train)
    wandb.sklearn.plot_summary_metrics(regressor, X_train, y_train, X_test, y_test)
    run = wandb.run
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("XGBoost model has a coefficient R^2 of %.3f on test data.", score)
    to_log = {
            "score":score
              }
    run.log(to_log)
    run.finish()
