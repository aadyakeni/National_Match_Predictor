import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import pickle

# Load datasets
goalscorers_df = pd.read_csv('assets/goalscorers.csv')
results_df = pd.read_csv('assets/results.csv')
shootouts_df = pd.read_csv('assets/shootouts.csv')

# Extract necessary columns
results_df = results_df[['date', 'home_team', 'away_team', 'home_score', 'away_score']]

# Convert date to datetime
results_df['date'] = pd.to_datetime(results_df['date'])

# Calculate match outcome
results_df['outcome'] = results_df.apply(lambda row: 'home_win' if row['home_score'] > row['away_score'] else ('away_win' if row['home_score'] < row['away_score'] else 'draw'), axis=1)

# Combine unique team names from both home and away columns for fitting
all_teams = pd.concat([results_df['home_team'], results_df['away_team']]).unique()

# Encode categorical features
label_encoder_teams = LabelEncoder()
label_encoder_outcome = LabelEncoder()

label_encoder_teams.fit(all_teams)

results_df['home_team_encoded'] = label_encoder_teams.transform(results_df['home_team'])
results_df['away_team_encoded'] = label_encoder_teams.transform(results_df['away_team'])
results_df['outcome_encoded'] = label_encoder_outcome.fit_transform(results_df['outcome'])

# Define features and targets
X = results_df[['home_team_encoded', 'away_team_encoded']]
y_outcome = results_df['outcome_encoded']
y_home_goals = results_df['home_score']
y_away_goals = results_df['away_score']

# Check for and handle missing values
print("Checking for NaNs in target variables:")
print("NaNs in y_home_goals:", y_home_goals.isna().sum())
print("NaNs in y_away_goals:", y_away_goals.isna().sum())

# Drop rows with NaNs in target variables
results_df = results_df.dropna(subset=['home_score', 'away_score'])

# Recompute the targets after dropping NaNs
X = results_df[['home_team_encoded', 'away_team_encoded']]
y_outcome = results_df['outcome_encoded']
y_home_goals = results_df['home_score']
y_away_goals = results_df['away_score']

# Split data into training and testing sets
X_train, X_test, y_train_outcome, y_test_outcome = train_test_split(X, y_outcome, test_size=0.2, random_state=42)
_, _, y_train_home_goals, y_test_home_goals = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)
_, _, y_train_away_goals, y_test_away_goals = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)

# Train initial models
clf_outcome = RandomForestClassifier(random_state=42)
clf_outcome.fit(X_train, y_train_outcome)

reg_home_goals = RandomForestRegressor(random_state=42)
reg_home_goals.fit(X_train, y_train_home_goals)

reg_away_goals = RandomForestRegressor(random_state=42)
reg_away_goals.fit(X_train, y_train_away_goals)

# Evaluate initial models
y_pred_outcome = clf_outcome.predict(X_test)
print("Initial outcome model performance:")
print(classification_report(y_test_outcome, y_pred_outcome))

y_pred_home_goals = reg_home_goals.predict(X_test)
print("Initial home goals model performance:")
print("MSE:", mean_squared_error(y_test_home_goals, y_pred_home_goals))

y_pred_away_goals = reg_away_goals.predict(X_test)
print("Initial away goals model performance:")
print("MSE:", mean_squared_error(y_test_away_goals, y_pred_away_goals))

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV for outcome model
grid_search_outcome = GridSearchCV(estimator=clf_outcome, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_outcome.fit(X_train, y_train_outcome)

# Perform GridSearchCV for home goals model
grid_search_home_goals = GridSearchCV(estimator=reg_home_goals, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_home_goals.fit(X_train, y_train_home_goals)

# Perform GridSearchCV for away goals model
grid_search_away_goals = GridSearchCV(estimator=reg_away_goals, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_away_goals.fit(X_train, y_train_away_goals)

# Get the best models
best_clf_outcome = grid_search_outcome.best_estimator_
best_reg_home_goals = grid_search_home_goals.best_estimator_
best_reg_away_goals = grid_search_away_goals.best_estimator_

# Make predictions with the best models
y_pred_best_outcome = best_clf_outcome.predict(X_test)
y_pred_best_home_goals = best_reg_home_goals.predict(X_test)
y_pred_best_away_goals = best_reg_away_goals.predict(X_test)

# Evaluate the best models
print("Best outcome model performance:")
print(classification_report(y_test_outcome, y_pred_best_outcome))

print("Best home goals model performance:")
print("MSE:", mean_squared_error(y_test_home_goals, y_pred_best_home_goals))

print("Best away goals model performance:")
print("MSE:", mean_squared_error(y_test_away_goals, y_pred_best_away_goals))

# Save the best models and label encoders using pickle
with open('best_model_outcome.pkl', 'wb') as f:
    pickle.dump(best_clf_outcome, f)
with open('best_model_home_goals.pkl', 'wb') as f:
    pickle.dump(best_reg_home_goals, f)
with open('best_model_away_goals.pkl', 'wb') as f:
    pickle.dump(best_reg_away_goals, f)
with open('label_encoder_teams.pkl', 'wb') as f:
    pickle.dump(label_encoder_teams, f)
with open('label_encoder_outcome.pkl', 'wb') as f:
    pickle.dump(label_encoder_outcome, f)

print("Best models and label encoders saved.")
