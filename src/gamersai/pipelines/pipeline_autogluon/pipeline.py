"""
This is a boilerplate pipeline 'pipeline_autogluon'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_autogluon, evaluate_model_autogluon
from .nodes import split_data
from ..knn_pipeline.nodes import evaluate_knn_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X", "y"],
                name="split_autogluon_data_node",
            ),
            node(
                func=evaluate_knn_model,
                inputs=["knn_model", "X_train", "y_train", "X_test", "y_test"],
                outputs="int_4",
                name="evaluate_knn_model_node",
            ),
            node(
                func=train_autogluon,
                inputs=["int_4", "X", "y", "params:time_limit"],
                outputs="autogluon_model",
                name="train_autogluon_node",
            ),
            node(
                func=evaluate_model_autogluon,
                inputs=["autogluon_model", "X", "y"],
                outputs= "int_5",
                name="evaluate_autogluon_model_node",
            ),
        ]
    )
