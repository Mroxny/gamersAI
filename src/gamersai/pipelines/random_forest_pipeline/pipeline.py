"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model, split_data, train_model, cross_validate_model
from ..pipeline_autogluon.nodes import evaluate_model_autogluon

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
                func=cross_validate_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs=None,
                name="cross_validate_model_node",
            ),
            node(
                func=evaluate_model_autogluon,
                inputs=["autogluon_model", "X", "y"],
                outputs= "int_5",
                name="evaluate_autogluon_model_node",
            ),
            node(
                func=train_model,
                inputs=["int_5","X_train", "y_train", "X_test", "y_test"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_train", "y_train", "X_test", "y_test"],
                outputs="int_6",
                name="evaluate_model_node",
            ),
        ]
    )