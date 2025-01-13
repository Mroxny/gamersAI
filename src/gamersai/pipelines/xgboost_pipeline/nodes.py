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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from wandb.integration.xgboost import WandbCallback
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold


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
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gamersAI",
        name = "XGBoost",
        group=os.environ["WANDB_RUN_GROUP"],
        # track hyperparameters and run metadata
        config=model.get_params()
    )
    
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
    logger.info("XGBoost model has a coefficient R^2 of %.3f on test data.", score)
    to_log = {
            "mse" : mse,
            "mae" : mae,
            "R2 score":score
              }
    run.log(to_log)
    run.finish()
    

def cross_validate_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    Performs cross-validation for the XGBoost model and logs R² scores to WandB.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.
        parameters: Hyperparameters for the XGBoost model.
    """
    # Define the XGBoost model
    model = XGBRegressor(
        objective=parameters["objective"],
        n_estimators=parameters["n_estimators"],
        learning_rate=parameters["learning_rate"],
        max_depth=parameters["max_depth"]
    )

    # Initialize WandB
    run = wandb.init(
        project="gamersAI",
        name="XGBoost_CrossValidation",
        group=os.environ.get("WANDB_RUN_GROUP"),
        config=model.get_params()
    )

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=101)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    # Log R² scores chart
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b')
    plt.title("XGBoost Cross-Validation R² Scores")
    plt.xlabel("Fold")
    plt.ylabel("R² Score")
    plt.grid()
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.ylim(-0.1, 1.1)  # Adjust the y-axis for clarity

    # Log the plot to WandB
    wandb.log({"R² Scores Plot": wandb.Image(plt)})
    plt.close()

    # Log the results to the console
    logger = logging.getLogger(__name__)
    logger.info("XGBoost Cross-validation R² scores: %s", cv_scores)
    logger.info("XGBoost Mean R² score: %.3f", cv_scores.mean())

    # Finish WandB run
    run.finish()