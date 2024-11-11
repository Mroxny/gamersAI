"""
This is a boilerplate pipeline 'svr_pipeline'
generated using Kedro 0.19.9
"""
import logging
import pandas as pd
import wandb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import wandb.sklearn

def train_tree_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> DecisionTreeRegressor:
    """Trains a Support Vector Regressor (SVR) model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.

    Returns:
        Trained SVR model.
    """
    model = DecisionTreeRegressor(random_state=parameters["random_state_dt"]) #zmień parametry

    model.fit(X_train, y_train)

    
    #wandb.sklearn.plot_regressor(model=model, X_train=X_train, X_test=X_test,y_train=y_train,y_test=y_test)
    wandb.sklearn.plot_learning_curve(model=model, X=X_train,y=y_train)
    wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)
    
    return model

def evaluate_tree_model(
    regressor: DecisionTreeRegressor, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the SVR model.

    Args:
        regressor: Trained SVR model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    run = wandb.run
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("DT model has a coefficient R^2 of %.3f on test data.", score)
    to_log = {
            "name": "DT",
            "score":score
              }
    run.log(to_log)
