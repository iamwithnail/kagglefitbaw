import pandas as pd
df = pd.read_csv('fixtures.csv', parse_dates=[1])
df = df.dropna(subset=['League', 'Matchdate', 'Hometeam', 'Awayteam'])
