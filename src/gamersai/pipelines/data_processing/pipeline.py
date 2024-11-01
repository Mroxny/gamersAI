from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_games

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_games,
                inputs="games",
                outputs="preprocessed_games",
                name="preprocess_games_node",
            )
        ]
    )