import streamlit as st
import requests
import dtos
import json

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

def load_kedro_session():
    bootstrap_project(Path.cwd())
    session = KedroSession.create()
    return session

def display_pagination_horizontal(total_pages):
    page_input = st.text_input("To Page", str(st.session_state.current_page))
    if page_input.isdigit():
        page_number = int(page_input)
        if 1 <= page_number <= total_pages:
            st.session_state.current_page = page_number

    col1, col2, col3 = st.columns([2, 6, 1])
    
    with col1:
        if st.button("Previous") and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()
    
    with col3:
        if st.button("Next") and st.session_state.current_page < total_pages :
            st.session_state.current_page += 1
            st.rerun()
            
    with col2:
        if total_pages <= 7 :
            page_columns = st.columns(total_pages)
            for idx, page in enumerate(range(1, total_pages + 1)):
                with page_columns[idx]:
                    if idx + 1 == st.session_state.current_page :                        
                        st.markdown(f"<h3 style='color:red'>{st.session_state.current_page}</h3>", unsafe_allow_html=True)
                    else:
                        if st.button(str(page)):
                            st.session_state.current_page = page
                            st.rerun()
        
        else:
            lastPages = total_pages - 3
            firstPages = 4
            page_columns = st.columns(7)
            if st.session_state.current_page <= 4:
                for idx, page in enumerate(range(1, 7)):
                    with page_columns[idx]:
                        if idx + 1 == st.session_state.current_page: 
                            st.markdown(f"<h3 style='color:red'>{st.session_state.current_page}</h3>", unsafe_allow_html=True)
                        else:
                            # Użycie unikalnego identyfikatora przycisku
                            button_id = f"page_{page}_{idx}"
                            if st.button(str(page), key=button_id):  # Klucz identyfikujący przycisk
                                st.session_state.current_page = page
                                st.rerun()
                
                with page_columns[6]:
                    button_id = f"page_{total_pages}_last"  # Unikalny identyfikator dla ostatniego przycisku
                    if st.button(str(total_pages), key=button_id):
                        st.session_state.current_page = total_pages
                        st.rerun()

            elif st.session_state.current_page >= lastPages : 
                with page_columns[0]:
                    if st.button(str(1)):
                        st.session_state.current_page = 1
                        st.rerun()
                for idx, page in enumerate(range(total_pages -5, total_pages+1)):
                    with page_columns[idx+1]:
                        if page == st.session_state.current_page: 
                            st.markdown(f"<h3 style='color:red'>{st.session_state.current_page}</h3>", unsafe_allow_html=True)
                        else:
                            if st.button(str(page)):  # Klucz identyfikujący przycisk
                                st.session_state.current_page = page
                                st.rerun()
            else :
                with page_columns[0]:
                    if st.button(str(1)):
                        st.session_state.current_page = 1
                        st.rerun()

                with page_columns[6]:
                    if st.button(str(total_pages)):
                        st.session_state.current_page = total_pages
                        st.rerun()
                curr = st.session_state.current_page
                for idx, page in enumerate(range(curr -2, curr+3)):
                    with page_columns[idx+1]:
                        if page == st.session_state.current_page: 
                            st.markdown(f"<h3 style='color:red'>{st.session_state.current_page}</h3>", unsafe_allow_html=True)
                        else:
                            if st.button(str(page)):  # Klucz identyfikujący przycisk
                                st.session_state.current_page = page
                                st.rerun()

def display_game_details(game_details):
    st.title(game_details.get("Name", "N/A"))
    st.image(game_details.get("Header_image", ""), caption="Game Header")

    st.subheader("Details")
    st.markdown(f"""
    **Release Date:** {game_details.get("Release_date", "N/A")}
    **Price:** ${game_details.get("Price", "N/A")}
    **Estimated Owners:** {game_details.get("Estimated_owners", "N/A")}
    **Peak Concurrent Users:** {game_details.get("Peak_CCU", "N/A")}
    **Required Age:** {game_details.get("Required_age", "N/A")}
    """)

    st.subheader("Game Description")
    st.markdown(game_details.get("About_the_game", "N/A"))

    st.subheader("Platform Availability")
    st.markdown(f"""
    - Windows: {"✅" if game_details.get("Windows", "False") == "True" else "❌"}
    - Mac: {"✅" if game_details.get("Mac", "False") == "True" else "❌"}
    - Linux: {"✅" if game_details.get("Linux", "False") == "True" else "❌"}
    """)

    st.subheader("Additional Information")
    st.markdown(f"""
    **Developers:** {game_details.get("Developers", "N/A")}
    **Publishers:** {game_details.get("Publishers", "N/A")}
    **Categories:** {game_details.get("Categories", "N/A").replace(",", ", ")}
    **Genres:** {game_details.get("Genres", "N/A").replace(",", ", ")}
    **Tags:** {game_details.get("Tags", "N/A").replace(",", ", ")}
    """)

    st.subheader("Media")
    screenshots = game_details.get("Screenshots", "").split(',')
    for screenshot in screenshots:
        st.image(screenshot)

    if game_details.get("Movies"):
        st.video(game_details.get("Movies"))

def main():
    st.set_page_config(layout="wide")
    st.title("Game API Interface")
    st.markdown("Select an option below to interact with the API.")

    option = st.selectbox("Choose an option", ["Predict Estimated Owners", "Get Game by Name", "Get All Games", "Add Game", "Retrain model", "Delete Game"])

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
                json_data = json.dumps(data)
                response = requests.post("http://127.0.0.1:8000/predict-estimated-owners", json=data)
                
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
        games = []
        total_count = 1
                
        if  game_name.strip() != '' :
            try:
                response = requests.get(f"http://127.0.0.1:8000/game/{game_name}/totalCount")
                if response.status_code == 200:
                    response_total_count = response.json()
                    total_count = int(response_total_count['count'])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                    st.error(f"Request failed: {e}")

            items_per_page = 5
            total_pages = (total_count + items_per_page - 1) // items_per_page
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1

            display_pagination_horizontal(total_pages)
            try:
                response = requests.get(f"http://127.0.0.1:8000/game/{game_name}?per_page={items_per_page}&page={st.session_state.current_page}")
                if response.status_code == 200:
                    games = response.json()
                    for item in games:
                        display_game_details(item)                  
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    elif option == "Get All Games":
        st.subheader("Get All Games")
        #per_page = st.number_input("Games per page", min_value=1, step=1, value=10)
        #page = st.number_input("Page number", min_value=1, step=1, value=1)

        #Get TotalCount
        total_count = 0
        try:
            response = requests.get(f"http://127.0.0.1:8000/games/totalCount")
            if response.status_code == 200:
                response_total_count = response.json()
                total_count = int(response_total_count['count'])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
                st.error(f"Request failed: {e}")

        items_per_page = 5
        total_pages = (total_count + items_per_page - 1) // items_per_page
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        display_pagination_horizontal(total_pages)

        try:
            response = requests.get(f"http://127.0.0.1:8000/games/?per_page={items_per_page}&page={st.session_state.current_page}")
            if response.status_code == 200:
                games = response.json()
                for item in games:
                    display_game_details(item)
                #st.json(games)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
                st.error(f"Request failed: {e}")

    elif option == "Add Game":
        st.subheader("Add Game")
        game_data = {
            "Name": st.text_input("Name"),
            "Release_date": st.text_input("Release Date"),
            "Estimated_owners": st.text_input("Estimated Owners"),
            "Peak_CCU": st.number_input("Peak CCU", min_value=0, step=1),
            "Required_age": st.number_input("Required Age", min_value=0, step=1),
            "Price": st.number_input("Price", min_value=0.0, step=0.1),
            "DLC_count": st.number_input("DLC Count", min_value=0, step=1),
            "About_the_game": st.text_area("About the Game"),
            "Supported_languages": st.text_input("Supported Languages"),
            "Full_audio_languages": st.text_input("Full Audio Languages"),
            "Reviews": st.text_area("Reviews"),
            "Header_image": st.text_input("Header Image URL"),
            "Website": st.text_input("Website"),
            "Support_url": st.text_input("Support URL"),
            "Support_email": st.text_input("Support Email"),
            "Windows": st.number_input("Windows Support", min_value=0, max_value=1, step=1),
            "Mac": st.number_input("Mac Support", min_value=0, max_value=1, step=1),
            "Linux": st.number_input("Linux Support", min_value=0, max_value=1, step=1),
            "Metacritic_score": st.number_input("Metacritic Score", min_value=0.0, step=0.1),
            "Metacritic_url": st.text_input("Metacritic URL"),
            "User_score": st.number_input("User Score", min_value=0.0, step=0.1),
            "Positive": st.number_input("Positive Reviews", min_value=0, step=1),
            "Negative": st.number_input("Negative Reviews", min_value=0, step=1),
            "Score_rank": st.number_input("Score Rank", min_value=0, step=1),
            "Achievements": st.number_input("Achievements", min_value=0, step=1),
            "Recommendations": st.number_input("Recommendations", min_value=0, step=1),
            "Notes": st.text_area("Notes"),
            "Average_playtime_forever": st.number_input("Average Playtime Forever", min_value=0, step=1),
            "Average_playtime_two_weeks": st.number_input("Average Playtime Two Weeks", min_value=0, step=1),
            "Median_playtime_forever": st.number_input("Median Playtime Forever", min_value=0, step=1),
            "Median_playtime_two_weeks": st.number_input("Median Playtime Two Weeks", min_value=0, step=1),
            "Developers": st.text_input("Developers"),
            "Publishers": st.text_input("Publishers"),
            "Categories": st.text_input("Categories"),
            "Genres": st.text_input("Genres"),
            "Tags": st.text_input("Tags"),
            "Screenshots": st.text_input("Screenshots"),
            "Movies": st.text_input("Movies")
        }

        if st.button("Add Game"):
            try:
                # Convert Estimated Owners
                estimated_owners = game_data["Estimated_owners"]
                if "-" not in estimated_owners:
                    try:
                        estimated_owners = f"{int(estimated_owners) - 5000} - {int(estimated_owners) + 5000}"
                    except ValueError:
                        st.error("Invalid input for Estimated Owners. Please enter a valid range or single number.")
                        return
                game_data["Estimated_owners"] = estimated_owners

                game = dtos.GameDTO(**game_data)
                response = requests.post("http://127.0.0.1:8000/games/", json=game.dict())
                if response.status_code == 200:
                    st.success("Game added successfully!")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                print (e)
                st.error(f"Request failed: {e}")

    elif option == "Retrain model":
        if st.button("Retrain models"):
            with st.spinner("Retraining..."):
                try:
                    session = load_kedro_session()
                    session.run(pipeline_name="pipeline_autogluon")
                    st.success("Retraining completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during retraining: {e}")
    elif option == "Delete Game":
        st.subheader("Delete Game")
        game_name = st.text_input("Enter the game name to delete")

        if st.button("Delete Game"):
            try:
                response = requests.delete(f"http://127.0.0.1:8000/game/{game_name}")
                if response.status_code == 200:
                    st.success("Game deleted successfully!")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()