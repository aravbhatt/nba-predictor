from nba_api.stats.endpoints import scoreboardv2, leaguedashteamstats
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import pandas as pd

TEAM_NAME_MAP = {
    'Hawks': 'Atlanta Hawks',
    'Celtics': 'Boston Celtics',
    'Nets': 'Brooklyn Nets',
    'Hornets': 'Charlotte Hornets',
    'Bulls': 'Chicago Bulls',
    'Cavaliers': 'Cleveland Cavaliers',
    'Mavericks': 'Dallas Mavericks',
    'Nuggets': 'Denver Nuggets',
    'Pistons': 'Detroit Pistons',
    'Warriors': 'Golden State Warriors',
    'Rockets': 'Houston Rockets',
    'Pacers': 'Indiana Pacers',
    'Clippers': 'Los Angeles Clippers',
    'Lakers': 'Los Angeles Lakers',
    'Grizzlies': 'Memphis Grizzlies',
    'Heat': 'Miami Heat',
    'Bucks': 'Milwaukee Bucks',
    'Timberwolves': 'Minnesota Timberwolves',
    'Pelicans': 'New Orleans Pelicans',
    'Knicks': 'New York Knicks',
    'Thunder': 'Oklahoma City Thunder',
    'Magic': 'Orlando Magic',
    '76ers': 'Philadelphia 76ers',
    'Suns': 'Phoenix Suns',
    'Trail Blazers': 'Portland Trail Blazers',
    'Kings': 'Sacramento Kings',
    'Spurs': 'San Antonio Spurs',
    'Raptors': 'Toronto Raptors',
    'Jazz': 'Utah Jazz',
    'Wizards': 'Washington Wizards',
}

# Getting Data for Today's Games
today = datetime.now().strftime('%m/%d/%Y')
games_df = scoreboardv2.ScoreboardV2(game_date=today).get_data_frames()[1]

# Stats for this season
stats_raw = leaguedashteamstats.LeagueDashTeamStats(season='2023-24').get_data_frames()[0]
stats = stats_raw[['TEAM_NAME', 'W_PCT', 'PTS', 'REB', 'AST']].set_index('TEAM_NAME')

# Model train
X = stats[['PTS', 'REB', 'AST']]
y = (stats['W_PCT'] > 0.5).astype(int)
model = LogisticRegression()
model.fit(X, y)

# Rebuilding the matchups
matchups = {}
for _, row in games_df.iterrows():
    game_id = row['GAME_ID']
    team = row['TEAM_NAME']
    if game_id not in matchups:
        matchups[game_id] = []
    matchups[game_id].append(team)

# Prediction for each game
print("Predictions for Today's Games:\n")

for game_id, teams in matchups.items():
    if len(teams) != 2:
        continue

    team1, team2 = teams
    mapped1 = TEAM_NAME_MAP.get(team1)
    mapped2 = TEAM_NAME_MAP.get(team2)

    if not mapped1 or not mapped2:
        print(f"Missing mapping for: {team1} or {team2}")
        continue

    try:
        stats1 = stats.loc[mapped1]
        stats2 = stats.loc[mapped2]

        input1 = pd.DataFrame([[stats1['PTS'], stats1['REB'], stats1['AST']]], columns=['PTS', 'REB', 'AST'])
        input2 = pd.DataFrame([[stats2['PTS'], stats2['REB'], stats2['AST']]], columns=['PTS', 'REB', 'AST'])

        pred1 = model.predict(input1)[0]
        pred2 = model.predict(input2)[0]



        winner = team1 if pred1 >= pred2 else team2
        print(f"{mapped1} vs {mapped2} -> Predicted Winner: {winner}")
    except KeyError:
        print(f"Stats missing for: {mapped1} or {mapped2}")
