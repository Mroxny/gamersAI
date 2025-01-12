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
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gamersAI",
        name = "Autogluon",
        group=os.environ["WANDB_RUN_GROUP"],
    )
    training_data = pd.concat([X_train, y_train], axis=1)
    predictor = TabularPredictor(label=y_train.name).fit(training_data, time_limit=time_limit)
    predictor.delete_models(models_to_keep="best", dry_run=False)
    return predictor

def evaluate_model_autogluon(predictor: TabularPredictor, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the AutoGluon model."""

    
    training_data = pd.concat([X_test, y_test], axis=1)
    dictionary = predictor.evaluate(training_data)
    importances = predictor.feature_importance(training_data)
    leaderboardDiktionary = predictor.leaderboard(training_data)

    
    run = wandb.run
    logger = logging.getLogger(__name__)

    #logger.info(dictionary)
    #logger.info(leaderboardDiktionary)
    #logger.info(leaderboardDiktionary.columns)
    #logger.info(importances.columns)

    table_leaderboard = wandb.Table(columns=["model", "score_test", "score_val", "eval_metric"])
    for index, row in leaderboardDiktionary.iterrows():
        table_leaderboard.add_data(row['model'], row['score_test'], row['score_val'], row['eval_metric'])

    table_importances = wandb.Table(columns=["feature", "importance", "stddev", "p_value", "n", "p99_high", "p99_low"])
    for index, row in importances.iterrows():
        table_importances.add_data(index, row['importance'], row['stddev'], row['p_value'], 
                               row['n'], row['p99_high'], row['p99_low'])
    
    wandb.log({"leaderboard": table_leaderboard})
    wandb.log({"importances": table_importances})
    run.finish()
