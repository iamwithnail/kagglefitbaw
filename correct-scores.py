import sqlite3
import pandas as pd
from sqlalchemy import create_engine # database connection
engine  = create_engine("sqlite:///database.sqlite").raw_connection()
countries = pd.read_sql_query('SELECT * FROM Country;', engine)
countries.rename(columns={'id':'country_id', 'name':'Country'}, inplace=True)
countries.head()

leagues = pd.read_sql_query('SELECT * FROM League;', engine)
leagues.rename(columns={'id':'league_id', 'name':'League'}, inplace=True)
leagues

matches = pd.read_sql_query('SELECT * FROM Match where league_id = 1729 ;', engine)


print(matches.head())