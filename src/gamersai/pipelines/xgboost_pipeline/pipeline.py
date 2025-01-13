"""
This is a boilerplate pipeline 'xgboost_pipeline'
generated using Kedro 0.19.9
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_xgboost_model, evaluate_xgboost_model, cross_validate_xgboost_model
from ..random_forest_pipeline.nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=cross_validate_xgboost_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs=None,
                name="cross_validate_xgboost_model_node",
            ),
            node(
                func=train_xgboost_model,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:model_options"],
                outputs="xgboost_model",
                name="train_xgboost_model_node",
            ),
            node(
                func=evaluate_xgboost_model,
                inputs=["xgboost_model", "X_train", "y_train", "X_test", "y_test"],
                outputs= None,
                name="evaluate_xgboost_model_node",
            ),
        ]
    )