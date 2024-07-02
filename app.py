import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved models and label encoders using pickle
with open('best_model_outcome.pkl', 'rb') as f:
    model_outcome = pickle.load(f)
with open('best_model_home_goals.pkl', 'rb') as f:
    model_home_goals = pickle.load(f)
with open('best_model_away_goals.pkl', 'rb') as f:
    model_away_goals = pickle.load(f)
with open('label_encoder_teams.pkl', 'rb') as f:
    label_encoder_teams = pickle.load(f)
with open('label_encoder_outcome.pkl', 'rb') as f:
    label_encoder_outcome = pickle.load(f)

# Load results data for additional features
results_df = pd.read_csv('assets/results.csv')
results_df['date'] = pd.to_datetime(results_df['date'])

# Streamlit app
st.title("Football Match Outcome and Expected Goals Predictor")

# Get the list of teams and add a placeholder
teams = label_encoder_teams.classes_
teams_with_placeholder = ["Select a team..."] + list(teams)

# Create a format function to handle the placeholder
def format_team(team):
    return "Select a team..." if team == "Select a team..." else team

home_team = st.selectbox("Select Home Team", teams_with_placeholder, format_func=format_team)
away_team = st.selectbox("Select Away Team", teams_with_placeholder, format_func=format_team)

if home_team != "Select a team..." and away_team != "Select a team...":
    if st.button("Predict Outcome and Expected Goals"):
        # Encode the selected teams
        home_team_encoded = label_encoder_teams.transform([home_team])[0]
        away_team_encoded = label_encoder_teams.transform([away_team])[0]

        # Make predictions
        prediction_outcome = model_outcome.predict([[home_team_encoded, away_team_encoded]])
        outcome_encoded = prediction_outcome[0]
        outcome = label_encoder_outcome.inverse_transform([outcome_encoded])[0]

        prediction_home_goals = model_home_goals.predict([[home_team_encoded, away_team_encoded]])
        prediction_away_goals = model_away_goals.predict([[home_team_encoded, away_team_encoded]])


        st.write(f"The predicted outcome is: {outcome}")
        st.write(f"Expected goals for {home_team}: {prediction_home_goals[0]:.2f}")
        st.write(f"Expected goals for {away_team}: {prediction_away_goals[0]:.2f}")

        # Fetch historical match data
        historical_matches = results_df[(results_df['home_team'] == home_team) & (results_df['away_team'] == away_team)]
        if not historical_matches.empty:
            st.write(f"Historical Matches between {home_team} and {away_team}:")
            st.write(historical_matches[['date', 'home_team', 'home_score', 'away_team', 'away_score']])
        else:
            st.write("No historical data available for this match-up.")

else:
    st.write("Please select both home and away teams.")


