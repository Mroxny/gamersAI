"""
This is a boilerplate pipeline 'knn_pipeline'
generated using Kedro 0.19.9
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_knn_model, evaluate_knn_model
from ..data_science.nodes import split_data

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
                func=train_knn_model,
                inputs=["X_train", "y_train","X_test","y_test","params:model_options"],
                outputs="knn_model",
                name="train_knn_model_node",
            ),
            node(
                func=evaluate_knn_model,
                inputs=["knn_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_knn_model_node",
            ),
        ]
    )

