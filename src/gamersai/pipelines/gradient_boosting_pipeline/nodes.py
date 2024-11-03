import logging
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

def train_gradient_boosting_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    """Trains a Gradient Boosting Regressor model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.

    Returns:
        Trained Gradient Boosting model.
    """
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=101)
    model.fit(X_train, y_train)
    return model

def evaluate_gradient_boosting_model(
    regressor: GradientBoostingRegressor, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the Gradient Boosting model.

    Args:
        regressor: Trained Gradient Boosting model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Gradient Boosting model has a coefficient R^2 of %.3f on test data.", score)
