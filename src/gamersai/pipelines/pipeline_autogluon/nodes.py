"""
This is a boilerplate pipeline 'pipeline_autogluon'
generated using Kedro 0.19.9
"""
import os
import logging
import pandas as pd
import wandb
import wandb.sklearn

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from typing import Tuple
from autogluon.tabular import TabularPredictor

def split_data(data: pd.DataFrame, parameters: dict) -> Tuple:
    X = data.drop(parameters["features"], axis='columns')
    y = data['Estimated owners (avg)']

    return X, y

def train_autogluon(X_train: pd.DataFrame, y_train: pd.Series, time_limit: int):
    training_data = pd.concat([X_train, y_train], axis=1)
    predictor = TabularPredictor(label=y_train.name).fit(training_data, time_limit=time_limit)
    return predictor

def evaluate_model(predictor: TabularPredictor, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the AutoGluon model."""

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gamersAI",
        name = "DT",
        group=os.environ["WANDB_RUN_GROUP"],
    )
    wandb.sklearn.plot_learning_curve(model=predictor, X=X_test,y=y_test)
    #wandb.sklearn.plot_summary_metrics(predictor, X_test, y_test, X_test, y_test)
    wandb.sklearn.plot_residuals(predictor,X_test,y_test)
    df_seperated_X_numerical = X_test.select_dtypes(include=['number','boolean'])
    df_seperated_X_numerical = X_test.replace({False: 0,True: 1})
    wandb.sklearn.plot_outlier_candidates(predictor,df_seperated_X_numerical,y_test)
    run = wandb.run
    y_pred = predictor.predict(X_test)
    performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

    score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("DT model has a coefficient R^2 of %.3f on test data.", score)
    to_log = {
            "mse" : mse,
            "mae" : mae,
            "R2 score":score
              }
    run.log(to_log)
    run.finish()
