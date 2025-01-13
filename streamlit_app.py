import streamlit as st
import requests
import dtos

def main():
    st.title("Game API Interface")
    st.markdown("Select an option below to interact with the API.")

    option = st.selectbox("Choose an option", ["Predict Estimated Owners", "Get Game by Name", "Get All Games"])

    if option == "Predict Estimated Owners":
        st.subheader("Predict Estimated Owners")
        st.markdown("Enter the game features below and click **Predict** to get the estimated owners.")

        # --- Section 1: Basic Game Info ---
        st.subheader("Basic Game Info")
        Peak_CCU = st.number_input("Peak CCU", min_value=0, step=1, help="Maximum concurrent users at peak")
        Required_age = st.number_input("Required Age", min_value=0, step=1, help="Minimum required age to play the game")
        Price = st.number_input("Price (USD)", min_value=0.0, step=0.1, help="Game price in USD")
        DLC_count = st.number_input("DLC Count", min_value=0, step=1, help="Number of DLCs for this game")

        # --- Section 2: System Support ---
        st.subheader("System Support")
        Windows_radio = st.radio("Windows Support?", ("Yes", "No"))
        Windows = 1 if Windows_radio == "Yes" else 0

        Mac_radio = st.radio("Mac Support?", ("Yes", "No"))
        Mac = 1 if Mac_radio == "Yes" else 0

        Linux_radio = st.radio("Linux Support?", ("Yes", "No"))
        Linux = 1 if Linux_radio == "Yes" else 0

        # --- Section 3: Metacritic & Reviews ---
        st.subheader("Metacritic & Reviews")
        Metacritic_score = st.number_input("Metacritic Score", min_value=0.0, step=0.1, help="Metascore from Metacritic")
        Positive = st.number_input("Positive Reviews", min_value=0, step=1, help="Total number of positive reviews")
        Negative = st.number_input("Negative Reviews", min_value=0, step=1, help="Total number of negative reviews")
        Achievements = st.number_input("Achievements", min_value=0, step=1, help="Number of in-game achievements")
        Recommendations = st.number_input("Recommendations", min_value=0, step=1, help="Number of recommendations from users")

        # --- Section 4: Gameplay & Language ---
        st.subheader("Gameplay & Language")
        Median_playtime_forever = st.number_input("Median Playtime (Forever)", min_value=0, step=1, 
                                                  help="Median total playtime in minutes")

        English_radio = st.radio("English Language?", ("Yes", "No"))
        Is_English = 1 if English_radio == "Yes" else 0

        Spanish_radio = st.radio("Spanish Language?", ("Yes", "No"))
        Is_Spanish = 1 if Spanish_radio == "Yes" else 0

        German_radio = st.radio("German Language?", ("Yes", "No"))
        Is_German = 1 if German_radio == "Yes" else 0

        French_radio = st.radio("French Language?", ("Yes", "No"))
        Is_French = 1 if French_radio == "Yes" else 0

        # --- Section 5: Game Modes ---
        st.subheader("Game Modes")
        single_player_radio = st.radio("Single-player?", ("Yes", "No"))
        Is_Single_player = 1 if single_player_radio == "Yes" else 0

        multi_player_radio = st.radio("Multi-player?", ("Yes", "No"))
        Is_Multi_player = 1 if multi_player_radio == "Yes" else 0

        # --- Section 6: Genre ---
        st.subheader("Genre")
        Genre_1_encoded = st.number_input("Genre_1_encoded", min_value=0, step=1, 
                                          help="Numeric code representing the game's main genre")

        # --- Prediction Button ---
        if st.button("Predict"):
            # Gather all form data into a dictionary
            data = {
                "Peak_CCU": Peak_CCU,
                "Required_age": Required_age,
                "Price": Price,
                "DLC_count": DLC_count,
                "Windows": Windows,
                "Mac": Mac,
                "Linux": Linux,
                "Metacritic_score": Metacritic_score,
                "Positive": Positive,
                "Negative": Negative,
                "Achievements": Achievements,
                "Recommendations": Recommendations,
                "Median_playtime_forever": Median_playtime_forever,
                "Is_English": Is_English,
                "Is_Spanish": Is_Spanish,
                "Is_German": Is_German,
                "Is_French": Is_French,
                "Is_Single_player": Is_Single_player,
                "Is_Multi_player": Is_Multi_player,
                "Genre_1_encoded": Genre_1_encoded
            }

            try:
                # Make sure your FastAPI server is running at the given URL
                game = dtos.GameDTO(**data)
                response = requests.post("http://127.0.0.1:8000/predict-estimated-owners", json=game.dict())
                # Check the response
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Estimated owners (avg): {result['Estimated owners (avg)']}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    elif option == "Get Game by Name":
        st.subheader("Get Game by Name")
        game_name = st.text_input("Enter the game name")

        if st.button("Get Game"):
            try:
                response = requests.get(f"http://127.0.0.1:8000/game/{game_name}")
                if response.status_code == 200:
                    game = response.json()
                    st.json(game)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    elif option == "Get All Games":
        st.subheader("Get All Games")
        per_page = st.number_input("Games per page", min_value=1, step=1, value=10)
        page = st.number_input("Page number", min_value=1, step=1, value=1)

        if st.button("Get Games"):
            try:
                response = requests.get(f"http://127.0.0.1:8000/games/?per_page={per_page}&page={page}")
                if response.status_code == 200:
                    games = response.json()
                    st.json(games)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()
