from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_games, create_model_input_table

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_games,
                inputs="games",
                outputs="preprocessed_games",
                name="preprocess_games_node",
            ),

            # Just in case we need more inputs
            node(
                func=create_model_input_table,
                inputs=["preprocessed_games"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            )
        ]
    )