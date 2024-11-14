"""
This is a boilerplate pipeline 'gradient_boosting_pipeline'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_gradient_boosting_model, evaluate_gradient_boosting_model
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
                func=train_gradient_boosting_model,
                inputs=["X_train", "y_train","X_test","y_test","params:model_options"],
                outputs="gradient_boosting_model",
                name="train_gradient_boosting_model_node",
            ),
            node(
                func=evaluate_gradient_boosting_model,
                inputs=["gradient_boosting_model", "X_train", "y_train","X_test","y_test"],
                outputs=None,
                name="evaluate_gradient_boosting_model_node",
            ),
        ]
    )
