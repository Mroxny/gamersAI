"""
This is a boilerplate pipeline 'xgboost_pipeline'
generated using Kedro 0.19.9
"""
import logging
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Trains the XGBoost regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.

    Returns:
        Trained XGBoost model.
    """
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    return model

def evaluate_xgboost_model(
    regressor: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the XGBoost model.

    Args:
        regressor: Trained XGBoost model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("XGBoost model has a coefficient R^2 of %.3f on test data.", score)
