"""
This is a boilerplate pipeline 'svr_pipeline'
generated using Kedro 0.19.9
"""
import logging
import pandas as pd
import wandb
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import wandb.sklearn

def train_svr_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> SVR:
    """Trains a Support Vector Regressor (SVR) model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.

    Returns:
        Trained SVR model.
    """
    model = SVR(kernel=parameters["kernel"], C=parameters["C"], epsilon=parameters["epsilon"])

    model.fit(X_train, y_train)

    # run = wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="gamersAI",

    #     # track hyperparameters and run metadata
    #     config=model.get_params()
    # )
    # #wandb.sklearn.plot_regressor(model=model, X_train=X_train, X_test=X_test,y_train=y_train,y_test=y_test)
    # wandb.sklearn.plot_learning_curve(model=model, X=X_train,y=y_train)
    #run.finish()
    return model

def evaluate_svr_model(
    regressor: SVR, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the SVR model.

    Args:
        regressor: Trained SVR model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("SVR model has a coefficient R^2 of %.3f on test data.", score)
