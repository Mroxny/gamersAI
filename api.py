from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

model_path = "api_model\model.pkl" 
with open(model_path, "rb") as file:
    model = pickle.load(file)

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

    input_data = pd.DataFrame([features.model_dump()])

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
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

