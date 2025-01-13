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
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

def train_knn_model( X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> KNeighborsRegressor:
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
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="gamersAI",
        name = "KNN",
        group=os.environ["WANDB_RUN_GROUP"],
        # track hyperparameters and run metadata
        config=model.get_params()
    )
    
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
    logger.info("KNN model has a coefficient R^2 of %.3f on test data.", score)
    to_log = {
            "mse" : mse,
            "mae" : mae,
            "R2 score":score
              }
    run.log(to_log)
    run.finish()

def cross_validate_knn_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    Performs cross-validation for the K-Nearest Neighbors model and logs the R² scores to WandB.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.
        parameters: Hyperparameters for the KNN model.
    """
    # Define the KNN model
    model = KNeighborsRegressor(
        n_neighbors=parameters["n_neighbors"]
    )

    # Initialize WandB
    run = wandb.init(
        project="gamersAI",
        name="KNN_CrossValidation",
        group=os.environ.get("WANDB_RUN_GROUP"),
        config=model.get_params()
    )

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=101)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    # Log R2 scores chart
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b')
    plt.title("KNN Cross-Validation R2 Scores")
    plt.xlabel("Fold")
    plt.ylabel("R2 Score")
    plt.grid()
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.ylim(-0.1, 1.1) 

    # Log the plot to WandB
    wandb.log({"R2 Scores Plot": wandb.Image(plt)})
    plt.close()

    # Log the results to the console
    logger = logging.getLogger(__name__)
    logger.info("KNN Cross-validation R2 scores: %s", cv_scores)
    logger.info("KNN Mean R2 score: %.3f", cv_scores.mean())

    # Finish WandB run
    run.finish()