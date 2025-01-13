from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import pandas as pd
import db_controller as dbc
from dtos import GameDTO
import traceback
import os

def find_model_by_prefix(path="AutogluonModels", prefix="ag"):
    if os.path.exists(path):
        for folder_name in os.listdir(path):
            # Check if folder starts with the prefix and is a directory
            folder_path = os.path.join(path, folder_name)
            if folder_name.startswith(prefix) and os.path.isdir(folder_path):
                # Construct the full path to the target file
                model_file_path = os.path.join(folder_path, "models", "WeightedEnsemble_L2", "model.pkl")
                if os.path.exists(model_file_path):
                    print(f"Model file found: {model_file_path}")
                    return model_file_path
                else:
                    print(f"'model.pkl' not found in {os.path.join(folder_path, 'models', 'WeightedEnsemble_L2')}")
                    return None
        print(f"No folder with prefix '{prefix}' found in '{path}'.")
    else:
        print(f"Path '{path}' does not exist.")
    return None


class GameFeatures(BaseModel):
    Peak_CCU: int
    Required_age: int
    Price: float
    DLC_count: int
    Windows: int
    Mac: int
    Linux: int
    Metacritic_score: float
    Positive: int
    Negative: int
    Achievements: int
    Recommendations: int
    Median_playtime_forever: int
    Is_English: int
    Is_Spanish: int
    Is_German: int
    Is_French: int
    Is_Single_player: int
    Is_Multi_player: int
    Genre_1_encoded: int

app = FastAPI()

@app.post("/predict-estimated-owners")
def predict_estimated_owners(features: GameFeatures):
    model_path = find_model_by_prefix()
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    input_data = pd.DataFrame([features.model_dump(mode='json')])

    expected_columns = [
        "Peak CCU", "Required age", "Price", "DLC count", "Windows", "Mac",
        "Linux", "Metacritic score", "Positive", "Negative", "Achievements",
        "Recommendations", "Median playtime forever", "Is English",
        "Is Spanish", "Is German", "Is French", "Is Single-player",
        "Is Multi-player", "Genre 1_encoded"
    ]

    input_data.rename(columns={
        "Peak_CCU": "Peak CCU",
        "Required_age": "Required age",
        "DLC_count": "DLC count",
        "Metacritic_score": "Metacritic score",
        "Median_playtime_forever": "Median playtime forever",
        "Is_English": "Is English",
        "Is_Spanish": "Is Spanish",
        "Is_German": "Is German",
        "Is_French": "Is French",
        "Is_Single_player": "Is Single-player",
        "Is_Multi_player": "Is Multi-player",
        "Genre_1_encoded": "Genre 1_encoded"
    }, inplace=True)

    input_data = input_data[expected_columns]

    try:
        prediction = model.predict(input_data)
        predicted_value = float(prediction[0])

        response_mapping = {
            0: 0,
            1: 10000,
            2: 45000,
            3: 100000,
            4: 200000,
            5: 450000
        }
        
        response = response_mapping.get(predicted_value, "Unknown prediction value")
        return {"Estimated owners (avg)": response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    
@app.get("/game/{name}")
async def get_game(name: str, per_page: int = Query(10, gt=0), page: int = Query(1, gt=0)):    
    offset = (page - 1) * per_page
    games = dbc.get_game_by_name(name,per_page, offset)
    if not games:
        raise HTTPException(status_code=404, detail="Game not found")
    return [dict(zip(GameDTO.model_fields.keys(), game[1:])) for game in games]

@app.get("/games/")
async def get_all_games(per_page: int = Query(10, gt=0), page: int = Query(1, gt=0)):
    """Get all games with pagination."""
    offset = (page - 1) * per_page
    games = dbc.get_all_games(per_page, offset)
    return [dict(zip(GameDTO.model_fields.keys(), game[1:])) for game in games]

@app.get("/games/totalCount")
async def get_total_count_games_async ():
    total_count = dbc.get_total_count_games()
    return {"count" :total_count[0]}

@app.get("/game/{name}/totalCount")
async def get_game(name: str):
    total_count = dbc.get_game_by_name_total_count(name)
    print(total_count)
    return {"count" :total_count[0]}

@app.post("/games/")
async def add_game(game: GameDTO):
    """Add a new game feature."""
    try:
        dbc.add_game(game)
        return {"message": "Feature added successfully"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

