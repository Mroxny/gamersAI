"""
This is a boilerplate pipeline 'svr_pipeline'
generated using Kedro 0.19.9
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_tree_model, evaluate_tree_model
from ..random_forest_pipeline.nodes import split_data  #..data_science.nodes

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
                func=train_tree_model,
                inputs=["X_train", "y_train","X_test","y_test","params:model_options"],
                outputs="dt_model",
                name="train_tree_model_node",
            ),
            node(
                func=evaluate_tree_model,
                inputs=["dt_model", "X_train", "y_train","X_test","y_test"],
                outputs=None,
                name="evaluate_tee_model_node",
            ),
        ]
    )
