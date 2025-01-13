import os
import pickle
import requests
import streamlit as st
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import pandas as pd 

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "417422402cmsh6e6ff0c00a5cb3fp1ca598jsn937b61106ee8")
RAPIDAPI_HOST = "nfl-api-data.p.rapidapi.com"


# Helper functions for Streamlit formatting
def to_markdown(text):
    text = text.replace('\u2022', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def load_model(model_path):
    """Load the trained model from a pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load models
nfl_winner_model = load_model('models/nfl_winner_model.pkl')
nba_player_model = load_model('models/linear_regression_model_player.pkl')
nhl_model = load_model('models/best_rf_model.pkl')

# NHL Prediction Functions
def fetch_live_scores_nhl():
    url = "https://api-web.nhle.com/v1/score/now"
    try:
        response = requests.get(url)
        response.raise_for_status()
        live_scores = response.json()
        live_game_data = []
        for game in live_scores.get('games', []):
            home_team = game['homeTeam']['abbrev']
            away_team = game['awayTeam']['abbrev']
            home_score = game['homeTeam'].get('score', 0)
            away_score = game['awayTeam'].get('score', 0)
            live_game_data.append({
                'Home_Team': home_team,
                'Away_Team': away_team,
                'Home_Score': home_score,
                'Away_Score': away_score,
            })
        return pd.DataFrame(live_game_data)
    except requests.exceptions.RequestException:
        return pd.DataFrame()

def preprocess_live_scores_nhl(live_scores, trained_features):
    """Preprocess live game data to match the trained model's feature set."""
    if live_scores.empty:
        return pd.DataFrame(columns=trained_features)
    live_scores.fillna(0, inplace=True)
    for col in trained_features:
        if col not in live_scores.columns:
            live_scores[col] = 0
    return live_scores[trained_features]

# NFL Prediction Functions
def fetch_live_scores_nfl():
    url = f"https://{RAPIDAPI_HOST}/nfl-livescores"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching live scores: {e}")
        return None

def predict_winner_nfl(home_team, away_team):
    """Predict winner between home and away team."""
    teams = [home_team, away_team]
    prediction = nfl_winner_model.predict([teams]) 
    return "Home" if prediction == 1 else "Away"

# NBA Prediction Functions
def get_gemini_response(question):
    """Fetch response from Gemini model."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

def predict_winner_nba(home_team, away_team):
    """Predict winner between home and away team using Gemini model."""
    prompt = f"Based on NBA historical data and team performance, who is likely to win: {home_team} (home) vs {away_team} (away)?"
    return get_gemini_response(prompt)

def player_points_prediction_nba(player_name):
    """Get points prediction for the player using Gemini model."""
    prompt = (f"Provide a detailed analysis of {player_name}'s performance, including points scored in the 2022-2023 and 2023-2024 seasons, "
              "and the current season 2024-2025. Include a brief analysis of trends and expectations.")
    return get_gemini_response(prompt)

def predict_mvp_nba():
    """Predict the most likely MVP for the current NBA season using Gemini model."""
    prompt = "Based on current NBA performance and historical trends, who is most likely to win the MVP award for the 2024-2025 season?"
    return get_gemini_response(prompt)

# Streamlit App
def main():
    st.set_page_config(page_title="Sports Prediction Models", layout="wide")
    st.title("Sports Prediction Models")

    # Choose sport
    sport = st.selectbox("Choose Sport", options=["NHL", "NFL", "NBA"])

    if sport == "NHL":
        # NHL Prediction Section
        st.subheader("NHL Prediction")
        live_scores = fetch_live_scores_nhl()
        if live_scores.empty:
            st.warning("No live games available.")
        else:
            st.write("Live Scores:", live_scores)

            trained_features = nhl_model.feature_names_in_
            processed_scores = preprocess_live_scores_nhl(live_scores, trained_features)
            predictions = nhl_model.predict(processed_scores)

            live_scores['Predicted Winner'] = [
                row['Home_Team'] if pred == 1 else row['Away_Team']
                for row, pred in zip(live_scores.to_dict('records'), predictions)
            ]
            st.write("Predictions:", live_scores[["Home_Team", "Away_Team", "Predicted Winner"]])

    elif sport == "NFL":
        # NFL Prediction Section
        st.subheader("NFL Prediction")
        live_scores = fetch_live_scores_nfl()
        if live_scores:
            st.write("Live Scores:", live_scores)

        teams = [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
            "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
            "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
            "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
            "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
            "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
            "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"
        ]

        team1 = st.selectbox("Select Team 1", options=teams)
        team2 = st.selectbox("Select Team 2", options=teams)

        if team1 == team2:
            st.warning("Please select two different teams.")
        else:
            if st.button("Predict Winner"):
                winner = predict_winner_nfl(team1, team2)
                st.success(f"The predicted winner is: **{winner}**")

    elif sport == "NBA":
        # NBA Prediction Section
        st.subheader("NBA Prediction")

        options = ["Winner Prediction", "Player Points Prediction", "MVP Prediction"]
        choice = st.radio("Select a feature:", options)

        if choice == "Winner Prediction":
            home_team = st.text_input("Enter Home Team:")
            away_team = st.text_input("Enter Away Team:")
            if st.button("Predict Winner"):
                if home_team and away_team:
                    result = predict_winner_nba(home_team, away_team)
                    st.write(f"Prediction: {result}")
                else:
                    st.error("Please enter both home and away team names.")

        elif choice == "Player Points Prediction":
            player_name = st.text_input("Enter Player Name:")
            if st.button("Predict Points"):
                if player_name:
                    points_result = player_points_prediction_nba(player_name)
                    st.write(f"Prediction: {points_result}")
                else:
                    st.error("Please enter a player name.")

        elif choice == "MVP Prediction":
            if st.button("Predict MVP"):
                mvp_result = predict_mvp_nba()
                st.write(f"Prediction: {mvp_result}")

if __name__ == "__main__":
    main()
