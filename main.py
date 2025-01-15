import os
import pickle
import requests
import streamlit as st
import textwrap

import numpy as np
import pandas as pd
from groq import Groq

# Load environment variables
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "417422402cmsh6e6ff0c00a5cb3fp1ca598jsn937b61106ee8")
RAPIDAPI_HOST = "nfl-api-data.p.rapidapi.com"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Helper functions
def load_model(model_path):
    """Load the trained model from a pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def to_markdown(text):
    """Convert text to markdown format."""
    text = text.replace('\u2022', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Load models
nfl_winner_model = load_model('models/nfl_winner_model.pkl')
nba_player_model = load_model('models/linear_regression_model_player.pkl')
nhl_model = load_model('models/best_rf_model.pkl')

# NHL Functions
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
    """Preprocess live NHL game data."""
    if live_scores.empty:
        return pd.DataFrame(columns=trained_features)
    live_scores.fillna(0, inplace=True)
    for col in trained_features:
        if col not in live_scores.columns:
            live_scores[col] = 0
    return live_scores[trained_features]

# NFL Functions
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
    """Predict NFL game winner using GROQ API."""
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"Based on NFL historical data and team performance, who is likely to win: {home_team} (home) vs {away_team} (away)?"
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# NBA Functions
def get_groq_response(prompt):
    """Fetch response from GROQ model."""
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def predict_winner_nba(home_team, away_team):
    """Predict NBA game winner."""
    prompt = f"Based on NBA historical data and team performance, who is likely to win: {home_team} (home) vs {away_team} (away)?"
    return get_groq_response(prompt)

def player_points_prediction(player_name):
    """Predict NBA player performance."""
    prompt = (f"Provide an analysis of {player_name}'s performance in the last three NBA seasons, "
              "including trends and expectations.")
    return get_groq_response(prompt)

def predict_mvp():
    """Predict NBA MVP."""
    prompt = "Who is the most likely MVP for the current NBA season?"
    return get_groq_response(prompt)

# Streamlit App
def main():
    st.set_page_config(page_title="Sports Prediction Models", layout="wide")
    st.title("Sports Prediction Models")

    # Navigation
    sport = st.sidebar.selectbox("Choose Sport", ["NHL", "NFL", "NBA"])

    if sport == "NHL":
        st.subheader("NHL Predictions")
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
        st.subheader("NFL Predictions")
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
        team1 = st.selectbox("Select Home Team", teams)
        team2 = st.selectbox("Select Away Team", teams)
        if st.button("Predict Winner"):
            if team1 != team2:
                winner = predict_winner_nfl(team1, team2)
                st.success(f"The predicted winner is: **{winner}**")
            else:
                st.warning("Please select two different teams.")

    elif sport == "NBA":
        st.subheader("NBA Predictions")
        choice = st.radio("Choose a feature", ["Winner Prediction", "Player Performance", "MVP Prediction"])
        if choice == "Winner Prediction":
            home_team = st.text_input("Enter Home Team")
            away_team = st.text_input("Enter Away Team")
            if st.button("Predict Winner"):
                if home_team and away_team:
                    result = predict_winner_nba(home_team, away_team)
                    st.success(f"Prediction: {result}")
                else:
                    st.error("Enter both teams.")
        elif choice == "Player Performance":
            player = st.text_input("Enter Player Name")
            if st.button("Predict Player Performance"):
                if player:
                    result = player_points_prediction(player)
                    st.success(result)
                else:
                    st.error("Enter a player name.")
        elif choice == "MVP Prediction":
            if st.button("Predict MVP"):
                result = predict_mvp()
                st.success(result)

if __name__ == "__main__":
    main()

