# National_FootballMatch_Predictor
 The project predicts football match outcomes and expected goals using machine learning models and historical match data.

## [The Live Website](https://nationalmatchpredictor.streamlit.app/)

### Features

- **Match Outcome Prediction**: Predicts whether the home team will win, the away team will win, or if the match will end in a draw.
- **Expected Goals Calculation**: Provides expected goals for both home and away teams, giving an estimate of how many goals each team is likely to score.

### Data and Models

The project utilizes historical match data and player performance statistics to train RandomForest models for each prediction task. The data is processed and encoded to fit the models, and extensive hyperparameter tuning is performed using GridSearchCV to ensure the best possible performance. The models and label encoders are saved using pickle, making them easy to load and use in a Streamlit app for real-time predictions.


