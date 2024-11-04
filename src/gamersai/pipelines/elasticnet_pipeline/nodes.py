"""
This is a boilerplate pipeline 'elasticnet_pipeline'
generated using Kedro 0.19.9
"""
import logging
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

def train_elasticnet_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> ElasticNet:
    """Trains an ElasticNet regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for the target.

    Returns:
        Trained ElasticNet model.
    """
    #model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=101)
    model = ElasticNet(alpha=parameters["alpha"], l1_ratio=parameters["l1_ratio"], random_state=parameters["random_state_elasticnet"])
    
    model.fit(X_train, y_train)
    return model

def evaluate_elasticnet_model(
    regressor: ElasticNet, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination for the ElasticNet model.

    Args:
        regressor: Trained ElasticNet model.
        X_test: Testing data of independent features.
        y_test: Testing data for the target.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("ElasticNet model has a coefficient R^2 of %.3f on test data.", score)
