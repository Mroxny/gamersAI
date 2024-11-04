"""
This is a boilerplate pipeline 'knn_pipeline'
generated using Kedro 0.19.9
"""
import logging
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> KNeighborsRegressor:
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
    return model

def evaluate_knn_model(
    regressor: KNeighborsRegressor, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the KNN model.

    Args:
        regressor: Trained K-Nearest Neighbors model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("K-Nearest Neighbors model has a coefficient R^2 of %.3f on test data.", score)
