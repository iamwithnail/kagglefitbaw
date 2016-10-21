#https://www.kaggle.com/petehodge/d/hugomathien/soccer/predicting-epl-scores-for-fun/discussion
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import sqlalchemy
from sqlalchemy import create_engine # database connection

from IPython.display import display, clear_output
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls",]).decode("utf8"))

# Any results you write to the current directory are saved as output.




engine  = create_engine("sqlite:///database v2.sqlite").raw_connection()
countries = pd.read_sql_query('SELECT * FROM Country;', engine)
countries.rename(columns={'id':'country_id', 'name':'Country'}, inplace=True)
countries.head()

leagues = pd.read_sql_query('SELECT * FROM League;', engine)
leagues.rename(columns={'id':'league_id', 'name':'League'}, inplace=True)
leagues

# Select a number of seasons in a list or just everything
#matches = pd.read_sql_query('SELECT * FROM Match where league_id = 1729 and season in ("2010/2011", "2011/2012", "2012/2013", "2013/2014", "2014/2015", "2015/2016");'
#                                          , engine)
#                           "2008/2009", "2009/2010",         \
matches = pd.read_sql_query('SELECT * FROM Match where league_id = 1729 ;', engine)

#matches.info()
print(matches.head())
# matches.tail()
#matches.shape
matches.describe
# matches.dtypes
#matches.loc[(matches["season"]=="2012/2013")].count()



matches = matches[matches.columns[:11]]
# sample = matches.league_id == 1729
#matches.dtypes

teams = pd.read_sql_query('SELECT * FROM Team;', engine)
print(teams.head())
#teams.loc[teams['team_long_name'] == 'Blackburn Rovers']

# Add home team name column
matches = pd.merge(left=matches, right=teams, how='left', left_on='home_team_api_id', right_on='team_api_id')
matches = matches.drop(['country_id','league_id', 'home_team_api_id', 'id_y', 'team_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)
matches.rename(columns={'id_x':'match_id', 'team_long_name':'home_team'}, inplace=True)
#matches.tail()

matches = pd.merge(left=matches, right=teams, how='left', left_on='away_team_api_id', right_on='team_api_id')
matches = matches.drop(['id', 'match_api_id', 'away_team_api_id','team_api_id', 'team_fifa_api_id', 'team_short_name'], axis=1)
matches.rename(columns={'id_x':'match_id', 'team_long_name':'away_team'}, inplace=True)
matches



#check_matches = matches.loc[((matches["home_team"]=="Arsenal") | (matches["away_team"]=="Arsenal"))
#& (matches["season"]=="2011/2012")].count()
#check_matches
# matches[:-1]
#matches.loc[matches['home_team'] == 'West Ham']
#unique_teams.sort_values('team')


# Add in this season (16/17) matches
latest_match_data = [
{'match_id':6000, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Swansea City'},
{'match_id':6001, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'West Bromwich Albion'},
{'match_id':6002, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Tottenham Hotspur'},
{'match_id':6003, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Leicester City'},
{'match_id':6004, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'Sunderland'},
{'match_id':6005, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Middlesbrough', 'away_team':'Stoke City'},
{'match_id':6006, 'season':'2016/2017', 'stage':1, 'date':'42595', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Watford'},
{'match_id':6007, 'season':'2016/2017', 'stage':1, 'date':'42596', 'home_team_goal':3, 'away_team_goal':4, 'home_team':'Arsenal', 'away_team':'Liverpool'},
{'match_id':6008, 'season':'2016/2017', 'stage':1, 'date':'42596', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Bournemouth', 'away_team':'Manchester United'},
{'match_id':6009, 'season':'2016/2017', 'stage':1, 'date':'42597', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Chelsea', 'away_team':'West Ham United'},
{'match_id':6010, 'season':'2016/2017', 'stage':2, 'date':'42601', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Manchester United', 'away_team':'Southampton'},
{'match_id':6011, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Liverpool'},
{'match_id':6012, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Arsenal'},
{'match_id':6013, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'Stoke City', 'away_team':'Manchester City'},
{'match_id':6014, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Hull City'},
{'match_id':6015, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Crystal Palace'},
{'match_id':6016, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Watford', 'away_team':'Chelsea'},
{'match_id':6017, 'season':'2016/2017', 'stage':2, 'date':'42602', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'Everton'},
{'match_id':6018, 'season':'2016/2017', 'stage':2, 'date':'42603', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Sunderland', 'away_team':'Middlesbrough'},
{'match_id':6019, 'season':'2016/2017', 'stage':2, 'date':'42603', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'West Ham United', 'away_team':'Bournemouth'},
{'match_id':6020, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Burnley'},
{'match_id':6021, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Bournemouth'},
{'match_id':6022, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Everton', 'away_team':'Stoke City'},
{'match_id':6023, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Hull City', 'away_team':'Manchester United'},
{'match_id':6024, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Leicester City', 'away_team':'Swansea City'},
{'match_id':6025, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Southampton', 'away_team':'Sunderland'},
{'match_id':6026, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Tottenham Hotspur', 'away_team':'Liverpool'},
{'match_id':6027, 'season':'2016/2017', 'stage':3, 'date':'42609', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Watford', 'away_team':'Arsenal'},
{'match_id':6028, 'season':'2016/2017', 'stage':3, 'date':'42610', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Manchester City', 'away_team':'West Ham United'},
{'match_id':6029, 'season':'2016/2017', 'stage':3, 'date':'42610', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Middlesbrough'},
{'match_id':6030, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':2, 'away_team_goal':1, 'home_team':'Arsenal', 'away_team':'Southampton'},
{'match_id':6031, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'West Bromwich Albion'},
{'match_id':6032, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Hull City'},
{'match_id':6033, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Leicester City'},
{'match_id':6034, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Manchester United', 'away_team':'Manchester City'},
{'match_id':6035, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Middlesbrough', 'away_team':'Crystal Palace'},
{'match_id':6036, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':0, 'away_team_goal':4, 'home_team':'Stoke City', 'away_team':'Tottenham Hotspur'},
{'match_id':6037, 'season':'2016/2017', 'stage':4, 'date':'42623', 'home_team_goal':2, 'away_team_goal':4, 'home_team':'West Ham United', 'away_team':'Watford'},
{'match_id':6038, 'season':'2016/2017', 'stage':4, 'date':'42624', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Chelsea'},
{'match_id':6039, 'season':'2016/2017', 'stage':4, 'date':'42625', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'Sunderland', 'away_team':'Everton'},
{'match_id':6040, 'season':'2016/2017', 'stage':5, 'date':'42629', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Chelsea', 'away_team':'Liverpool'},
{'match_id':6041, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Middlesbrough'},
{'match_id':6042, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':1, 'away_team_goal':4, 'home_team':'Hull City', 'away_team':'Arsenal'},
{'match_id':6043, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Burnley'},
{'match_id':6044, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':4, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Bournemouth'},
{'match_id':6045, 'season':'2016/2017', 'stage':5, 'date':'42630', 'home_team_goal':4, 'away_team_goal':2, 'home_team':'West Bromwich Albion', 'away_team':'West Ham United'},
{'match_id':6046, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Crystal Palace', 'away_team':'Stoke City'},
{'match_id':6047, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Swansea City'},
{'match_id':6048, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Sunderland'},
{'match_id':6049, 'season':'2016/2017', 'stage':5, 'date':'42631', 'home_team_goal':3, 'away_team_goal':1, 'home_team':'Watford', 'away_team':'Manchester United'},
{'match_id':6050, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':3, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Chelsea'},
{'match_id':6051, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Everton'},
{'match_id':6052, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':5, 'away_team_goal':1, 'home_team':'Liverpool', 'away_team':'Hull City'},
{'match_id':6053, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':4, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Leicester City'},
{'match_id':6054, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Middlesbrough', 'away_team':'Tottenham Hotspur'},
{'match_id':6055, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Stoke City', 'away_team':'West Bromwich Albion'},
{'match_id':6056, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':2, 'away_team_goal':3, 'home_team':'Sunderland', 'away_team':'Crystal Palace'},
{'match_id':6057, 'season':'2016/2017', 'stage':6, 'date':'42637', 'home_team_goal':1, 'away_team_goal':3, 'home_team':'Swansea City', 'away_team':'Manchester City'},
{'match_id':6058, 'season':'2016/2017', 'stage':6, 'date':'42638', 'home_team_goal':0, 'away_team_goal':3, 'home_team':'West Ham United', 'away_team':'Southampton'},
{'match_id':6059, 'season':'2016/2017', 'stage':6, 'date':'42639', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Burnley', 'away_team':'Watford'},
{'match_id':6060, 'season':'2016/2017', 'stage':7, 'date':'42643', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Everton', 'away_team':'Crystal Palace'},
{'match_id':6061, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':0, 'away_team_goal':2, 'home_team':'Hull City', 'away_team':'Chelsea'},
{'match_id':6062, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Sunderland', 'away_team':'West Bromwich Albion'},
{'match_id':6063, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':1, 'away_team_goal':2, 'home_team':'Swansea City', 'away_team':'Liverpool'},
{'match_id':6064, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':2, 'away_team_goal':2, 'home_team':'Watford', 'away_team':'Bournemouth'},
{'match_id':6065, 'season':'2016/2017', 'stage':7, 'date':'42644', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'West Ham United', 'away_team':'Middlesbrough'},
{'match_id':6066, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':0, 'away_team_goal':1, 'home_team':'Burnley', 'away_team':'Arsenal'},
{'match_id':6067, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Leicester City', 'away_team':'Southampton'},
{'match_id':6068, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':1, 'away_team_goal':1, 'home_team':'Manchester United', 'away_team':'Stoke City'},
{'match_id':6069, 'season':'2016/2017', 'stage':7, 'date':'42645', 'home_team_goal':2, 'away_team_goal':0, 'home_team':'Tottenham Hotspur', 'away_team':'Manchester City'},
{'match_id':6070, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Chelsea', 'away_team':'Leicester City'},
{'match_id':6071, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Arsenal', 'away_team':'Swansea City'},
{'match_id':6072, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Bournemouth', 'away_team':'Hull City'},
{'match_id':6073, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Manchester City', 'away_team':'Everton'},
{'match_id':6074, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Stoke City', 'away_team':'Sunderland'},
{'match_id':6075, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'West Bromwich Albion', 'away_team':'Tottenham Hotspur'},
{'match_id':6076, 'season':'2016/2017', 'stage':8, 'date':'42658', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Crystal Palace', 'away_team':'West Ham United'},
{'match_id':6077, 'season':'2016/2017', 'stage':8, 'date':'42659', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Middlesbrough', 'away_team':'Watford'},
{'match_id':6078, 'season':'2016/2017', 'stage':8, 'date':'42659', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Southampton', 'away_team':'Burnley'},
{'match_id':6079, 'season':'2016/2017', 'stage':8, 'date':'42660', 'home_team_goal':0, 'away_team_goal':0, 'home_team':'Liverpool', 'away_team':'Manchester United'}
]

latest_matches = pd.DataFrame(latest_match_data, columns=['match_id', 'season', 'stage', 'date',
                                                          'home_team_goal', 'away_team_goal',
                                                          'home_team','away_team'])
latest_matches

# Add to full training data to predict current season
#matches = pd.concat([matches, latest_matches])
#matches = matches.reset_index(drop=True)
#matches


full_matches = matches.copy()

# Regression model (see notes later in notebook)
# Build the initial model and update for each subsequent season
unique_seasons = pd.Series(matches['season'].unique())
regress_seasons = pd.Series(['2008/2009', '2009/2010', '2010/2011'])
regress_matches = full_matches.loc[full_matches['season'].isin(regress_seasons)]

remaining_seasons = unique_seasons[~unique_seasons.isin(regress_seasons)]

# Remove the regression seasons from the general training matches
#full_matches = full_matches.loc[full_matches['season'].isin(remaining_seasons)]
#full_matches.reset_index(drop=True, inplace=True)

# Optimum appears to be for season 10/11 onwards
exclude_seasons = pd.Series(['2008/2009', '2009/2010'])
include_seasons = unique_seasons[~unique_seasons.isin(exclude_seasons)]
full_matches = full_matches.loc[full_matches['season'].isin(include_seasons)]
#full_matches.reset_index(drop=True, inplace=True)

# Cope with newly promoted teams with limited or no stats
# Work out an average of newly promoted teams from matches in their first season and use that
# For new teams in the season I am predicting

###################################################################################
# 1. Identify newly promoted teams and their first season
# 2. Select all of their matches into a dataframe similar to matches
# Build a model specific to new teams
# 3. Aggregate their data as a single team - average performance of all new teams
# 4. This can be added to the main model as team 'Newly Promoted"
# 5. For prediction, any newly promoted team uses the new team average
###################################################################################

# For now, remove new teams from the test set
# new_teams = ['Watford', 'Bournemouth', 'Leicester']
# test_matches = test_matches[~test_matches['home_team'].isin(new_teams)]
# test_matches = test_matches[~test_matches['away_team'].isin(new_teams)]

# Newly promoted teams each season
team_data = {'team':['West Bromwich Albion', 'Stoke City', 'Hull City',
                     'Wolverhampton Wanderers', 'Birmingham City', 'Burnley',
                     'Newcastle United', 'West Bromwich Albion', 'Blackpool',
                     'Queens Park Rangers', 'Norwich City', 'Swansea City',
                     'Reading', 'Southampton', 'West Ham United',
                     'Cardiff City', 'Crystal Palace', 'Hull City',
                     'Leicester City', 'Burnley', 'Queens Park Rangers',
                     'Bournemouth', 'Watford', 'Norwich City',
                     'Burnley', 'Middlesbrough', 'Hull City'
                    ],
             'season':["2008/2009", "2008/2009", "2008/2009",
                       "2009/2010", "2009/2010", "2009/2010",
                       "2010/2011", "2010/2011", "2010/2011",
                       "2011/2012", "2011/2012", "2011/2012",
                       "2012/2013", "2012/2013", "2012/2013",
                       "2013/2014", "2013/2014", "2013/2014",
                       "2014/2015", "2014/2015", "2014/2015",
                       "2015/2016", "2015/2016", "2015/2016",
                       "2016/2017", "2016/2017", "2016/2017"
                      ]
            }
new_teams = pd.DataFrame(team_data, columns=['team', 'season'])

# new teams for "2016/2017" are 'Burnley', 'Middlesbrough', 'Hull City'

def create_new_team_matches(match_list, team_list):
    # Select all new team matches into a dataframe similar to matches
    new_team_match_list = pd.DataFrame()
    for index, row in team_list.iterrows():
        new_team_match_list = pd.concat([new_team_match_list, match_list.loc[((match_list['home_team'] == row['team']) |
                                                      (match_list['away_team'] == row['team'])) &
                                                       (match_list['season'] == row['season']) ]])

    # Remove duplicates by identifying index dupes and then using that as a filter
    new_team_match_list = new_team_match_list[~new_team_match_list.index.duplicated()]
    return new_team_match_list


new_team_matches = create_new_team_matches(matches, new_teams)
# Various ways to check that I have what I want:

# new_team_matches_agg = new_team_matches.groupby(['season', 'away_team']).count()
#new_team_matches
# new_team_matches.loc[new_team_matches['season'] == "2011/2012"]
# new_team_matches.loc[new_team_matches['match_id'] == 1897]



# Rename newly promoted teams to generic "Promoted"

for index, row in new_teams.iterrows():
    for index1, row1 in new_team_matches.iterrows():
        if (row1['home_team'] == row['team']) & (row1['season'] == row['season']):
            new_team_matches.loc[index1, 'home_team'] = 'Promoted'
        if (row1['away_team'] == row['team']) & (row1['season'] == row['season']):
            new_team_matches.loc[index1, 'away_team'] = 'Promoted'

#    new_team_matches = pd.concat([new_team_matches,matches.loc[((matches['home_team'] == row['team']) |
#                                                  (matches['away_team'] == row['team'])) &
#                                                   (matches['season'] == row['season']) ]])
new_team_matches

# Remove season 15/16 into a separate test set if testing against this
test_matches = matches[matches.season == "2015/2016"]
# If predicting this season (16/17), leave in 15/16 for the model - comment out
# matches = matches[matches.season != "2015/2016"]

# Set up the test matches, renaming new teams to 'Promoted' to use the new team percentages
# 15/16 new_teams = ['Watford', 'Bournemouth', 'Leicester']
# new teams for "2016/2017" are 'Burnley', 'Middlesbrough', 'Hull'
this_season_new_teams = pd.DataFrame({'team':['Watford', 'Bournemouth', 'Leicester City']})

for index, row in this_season_new_teams.iterrows():
    for index1, row1 in test_matches.iterrows():
        if row1['home_team'] == row['team']:
            test_matches.loc[index1, 'home_team'] = 'Promoted'
        if row1['away_team'] == row['team']:
            test_matches.loc[index1, 'away_team'] = 'Promoted'

test_matches = test_matches.reset_index(drop=True)
#test_matches
# this_season_new_teams



# Function to determine whether result is a win/draw/lose
# Passed the test_match dataframe, returns result

def determine_home_result(match):
    if match['home_team_goal'] > match['away_team_goal']:
        return 'win'
    elif match['home_team_goal'] < match['away_team_goal']:
        return 'lose'
    else:
        return 'draw'


# Set the home team result based on score. Will be used to compare predictions
test_matches['home_team_result'] = test_matches.apply(determine_home_result, axis=1)
test_matches
print(test_matches)


''' Included for info as a part of my journey
# First look to state the result as home or away win/draw/lose - 6 possibilites
# Here is an experiment with using a function and apply to get a single column with
# result but it didn't lend itself to further manipulation

# Functions to determine whether result is a win/draw/lose
# Passed the match dataframe, returns result

def determine_home_result(match):
    if match['home_team_goal'] > match['away_team_goal']:
        return 'win'
    elif match['home_team_goal'] < match['away_team_goal']:
        return 'loss'
    else:
        return 'draw'


def determine_away_result(match):
    if match['away_team_goal'] > match['home_team_goal']:
        return 'win'
    elif match['away_team_goal'] < match['home_team_goal']:
        return 'loss'
    else:
        return 'draw'


matches['home_team_result'] = matches.apply(determine_home_result, axis=1)
matches['away_team_result'] = matches.apply(determine_away_result, axis=1)
'''
# First look to state the result as home or away win/draw/lose - 6 possibilites
# Create a binary result
def determine_result(match_list):
    match_list['home_win'] = np.where(match_list['home_team_goal'] > match_list['away_team_goal'], 1, 0)
    match_list['home_draw'] = np.where(match_list['home_team_goal'] == match_list['away_team_goal'], 1, 0)
    match_list['home_lose'] = np.where(match_list['home_team_goal'] < match_list['away_team_goal'], 1, 0)
    match_list['away_win'] = np.where(match_list['home_team_goal'] < match_list['away_team_goal'], 1, 0)
    match_list['away_draw'] = np.where(match_list['home_team_goal'] == match_list['away_team_goal'], 1, 0)
    match_list['away_lose'] = np.where(match_list['home_team_goal'] > match_list['away_team_goal'], 1, 0)

print(determine_result(matches))
matches

''' Included for info as a part of my journey
Looked at various ways of getting these results, some of the code I fiddled with:
# Creating a team list first
team = matches.ix[:,7]
team.drop_duplicates(inplace=True)

# Count number of HW's where home_team is same as team name
# tmp2 = pd.value_counts(matches['home_team_result'] [])
# tmp2 = matches.groupby(['home_team','home_team_result'])['home_team_result'].count()

That finally lead to group by and aggregation
'''


# Next create a new dataframe with a row for team and totals for results:
# Team | Home Win | HD | HL | AW | AD | AL
def aggregate_team_results(match_list):
    # First aggregate the home matches
    team_res_agg = match_list.groupby(['home_team']).agg({'home_win': sum,
                                                          'home_draw': sum,
                                                          'home_lose': sum
                                                          })

    # Now append the away scores for the same team
    team_res_agg = pd.concat([team_res_agg,
                              match_list.groupby(['away_team']).agg({'away_win': sum,
                                                                     'away_draw': sum,
                                                                     'away_lose': sum
                                                                     })
                              ], axis=1).reset_index()

    # type(team_results_agg)
    team_res_agg.rename(columns={'index': 'team'}, inplace=True)
    return team_res_agg


team_results_agg = aggregate_team_results(matches)
print(team_results_agg)

# Now create the model for promoted teams and aggregate to a single team. Then add to
# the main model

# First look to state the result as home or away win/draw/lose - 6 possibilites
# Create a binary result

new_team_matches['home_win'] = np.where(new_team_matches['home_team_goal'] > new_team_matches['away_team_goal'], 1, 0)
new_team_matches['home_draw'] = np.where(new_team_matches['home_team_goal'] == new_team_matches['away_team_goal'], 1, 0)
new_team_matches['home_lose'] = np.where(new_team_matches['home_team_goal'] < new_team_matches['away_team_goal'], 1, 0)
new_team_matches['away_win'] = np.where(new_team_matches['home_team_goal'] < new_team_matches['away_team_goal'], 1, 0)
new_team_matches['away_draw'] = np.where(new_team_matches['home_team_goal'] == new_team_matches['away_team_goal'], 1, 0)
new_team_matches['away_lose'] = np.where(new_team_matches['home_team_goal'] > new_team_matches['away_team_goal'], 1, 0)

# Aggregate the home matches
new_team_results_agg = new_team_matches.groupby(['home_team']).agg({'home_win':sum,
                                                       'home_draw':sum,
                                                       'home_lose':sum
                                                      })

# Now append the away scores for the same team
new_team_results_agg = pd.concat([new_team_results_agg,
                              new_team_matches.groupby(['away_team']).agg({'away_win':sum,
                                                                  'away_draw':sum,
                                                                  'away_lose':sum
                                                                 })
                             ], axis=1).reset_index()

new_team_results_agg.rename(columns={'index':'team'}, inplace=True)

# Pull out the Promoted team row and add to main model
single_new_team_results_agg = new_team_results_agg.loc[new_team_results_agg['team'] == 'Promoted']
team_results_agg = pd.concat([team_results_agg, single_new_team_results_agg])
team_results_agg = team_results_agg.reset_index(drop=True)
print(team_results_agg)

# Now convert these absolute numbers into percentage of the total home or away matches
team_results_agg['home_win_pct'] = team_results_agg['home_win'] / (team_results_agg['home_win'] +
                                                                   team_results_agg['home_draw'] +
                                                                   team_results_agg['home_lose'])
team_results_agg['home_draw_pct'] = team_results_agg['home_draw'] / (team_results_agg['home_win'] +
                                                                   team_results_agg['home_draw'] +
                                                                   team_results_agg['home_lose'])
team_results_agg['home_lose_pct'] = team_results_agg['home_lose'] / (team_results_agg['home_win'] +
                                                                   team_results_agg['home_draw'] +
                                                                   team_results_agg['home_lose'])

team_results_agg['away_win_pct'] = team_results_agg['away_win'] / (team_results_agg['away_win'] +
                                                                   team_results_agg['away_draw'] +
                                                                   team_results_agg['away_lose'])
team_results_agg['away_draw_pct'] = team_results_agg['away_draw'] / (team_results_agg['away_win'] +
                                                                   team_results_agg['away_draw'] +
                                                                   team_results_agg['away_lose'])
team_results_agg['away_lose_pct'] = team_results_agg['away_lose'] / (team_results_agg['away_win'] +
                                                                   team_results_agg['away_draw'] +
                                                                   team_results_agg['away_lose'])

team_results_agg

team_results_agg_pct =  lambda x: x / team_results_agg.sum(axis=1)

print(team_results_agg_pct)




# Get a list of the matches for the season
predict_results = test_matches[['home_team', 'away_team']]


# new teams for "2016/2017" are 'Burnley', 'Middlesbrough', 'Hull'
# Or predict this week's matches
this_week_data = {'home_team':['Bournemouth', 'Arsenal', 'Promoted', 'Chelsea', 'Promoted',
                               'Leicester City', 'Liverpool', 'Manchester City', 'Swansea City', 'West Ham United'],
                   'away_team':['Tottenham Hotspur', 'Promoted', 'Everton', 'Manchester United', 'Stoke City',
                                'Crystal Palace', 'West Bromwich Albion', 'Southampton', 'Watford', 'Sunderland']}


this_week_matches = pd.DataFrame(this_week_data,
                                 columns=['home_team', 'away_team'])
this_week_matches
predict_results = this_week_matches[['home_team', 'away_team']]
print (predict_results)

home_results = team_results_agg[['team', 'home_win_pct',
                                  'home_draw_pct', 'home_lose_pct']]
home_results.columns=['team', 'win', 'draw', 'lose']



# Build dataframes for the away teams results model
away_results = team_results_agg[['team', 'away_win_pct',
                                  'away_draw_pct', 'away_lose_pct']]
away_results.columns=['team', 'win', 'draw', 'lose']

#away_results


# 1. Append W/D/L for home team
predict_results = pd.merge(left=predict_results, right=home_results, how='left', left_on='home_team', right_on='team')
predict_results = predict_results.drop(['team'], axis=1)
#matches.rename(columns={'id_x':'match_id', 'team_long_name':'home_team'}, inplace=True)
predict_results

# 2. Get W/D/L for away team
predict_results = pd.merge(left=predict_results, right=away_results, how='left', left_on='away_team', right_on='team')
predict_results = predict_results.drop(['team'], axis=1)

# 3. Take average of the two for - remember a home win is an away lose
predict_results['win'] = (predict_results['win_x'] + predict_results['lose_y']) / 2
predict_results['draw'] = (predict_results['draw_x'] + predict_results['draw_y']) / 2
predict_results['lose'] = (predict_results['lose_x'] + predict_results['win_y']) / 2
predict_results = predict_results.drop(['win_x', 'draw_x', 'lose_x', 'win_y', 'draw_y', 'lose_y'], axis=1)
#predict_results

# Function to determine whether the highest prediction is for win/draw/lose

def predict_home_result(match):
    if (match['win'] >= match['draw']) & (match['win'] >= match['lose']):
        return 'win' # Favour a home win if probability equal
    elif (match['lose'] > match['win']) & (match['lose'] > match['draw']):
        return 'lose'
    else:
        return 'draw'



predict_results['home_team_result'] = predict_results.apply(predict_home_result, axis=1)
print(predict_results)

# Get the actual result into the prediction table...

predict_results['actual_result'] = test_matches['home_team_result']

print("PREDICTED RESULTS")
print(predict_results)
print(predict_results)



# What result gets predicted better?
predict_correct = predict_results[['home_team_result', 'actual_result']][predict_results['actual_result'] == predict_results['home_team_result']]
predict_analysis = predict_correct.groupby('home_team_result').count()
predict_total = predict_results[['home_team_result', 'actual_result']]
predict_analysis['Total'] = predict_total.groupby('home_team_result').count()
predict_analysis['% Correct'] = predict_analysis['actual_result'] / predict_analysis['Total']

print(predict_analysis)
# predict_total
# predict_correct
# predict_win


actuals = predict_results[predict_results['actual_result'] == 'win']
print("Actuals")
print(actuals)



test_f = pd.DataFrame(full_matches['season'],columns=['season'])
#test_run_results = pd.DataFrame(data, columns=['Test Description','Summary Percentage'])
test_f = test_f.loc[test_f['season'] != '2015/2016']

print(test_f.shape)
print(full_matches.shape)

subset_matches = matches.loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')]
# matches['home_draw'].loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')]
# targets = np.array(matches['home_draw'].loc[(matches['home_team'] == 'Arsenal') & (matches['away_team'] == 'Liverpool')])
# targets

#test_matches['home_team_result'].values

# Convert home & team into a binary feature, ie Arsenal_h or Arsenal_a
match_features = pd.get_dummies(matches['home_team']).rename(columns=lambda x: str(x) + '_h')
match_features = pd.concat([match_features, pd.get_dummies(matches['away_team']).rename(columns=lambda x: str(x) + '_a')],
                         axis=1)
#match_features


# Cater for new teams by setting the new team for that season to a generic name
# Change home team names first, then away teams
def set_promoted_teams(new_team_list, match_list):
    for index, row in new_team_list.iterrows():
        for index1, row1 in match_list.iterrows():
            if (row1['home_team'] == row['team']) & (row1['season'] == row['season']):
                match_list.loc[index1, 'home_team'] = 'Promoted'
            if (row1['away_team'] == row['team']) & (row1['season'] == row['season']):
                match_list.loc[index1, 'away_team'] = 'Promoted'


# Create a regression type model for each season, for each team:
# Build a W/D/L average for a season, and as the new season progresses, subtract the results
# from the average model. So if a team normally has 18 wins in a season, and they have had 9,
# then their chances of more wins will reduce (as they regress to their normal pattern).
#
# First, build an average season for each team based on at least 3 seasons
# Then use this for the whole of the following season, recalculating the current regression
# after each match result.

# Build a regression model for each subsequent season after the first 3

regr_model_seasons = pd.DataFrame(regress_seasons, columns=['season'])
regr_model_remain_seasons = pd.DataFrame(remaining_seasons, columns=['season'])
regr_model_remain_seasons = regr_model_remain_seasons.reset_index(drop=True)
regr_model_team_results_agg = pd.DataFrame()

# Build a model for each season, then add to the main dataframe
for i, row in regr_model_remain_seasons.iterrows():
    # Select the relevant matches
    regr_model_matches = matches.loc[matches['season'].isin(regr_model_seasons['season'])]
    regr_model_this_seas_team_results_agg = aggregate_team_results(regr_model_matches)

    # Do the same for new teams
    regr_model_new_teams = new_teams.loc[new_teams['season'].isin(regr_model_seasons['season'])]
    regr_model_new_team_matches = create_new_team_matches(regr_model_matches, regr_model_new_teams)

    set_promoted_teams(regr_model_new_teams, regr_model_new_team_matches)
    regr_model_new_team_results_agg = aggregate_team_results(regr_model_new_team_matches)

    # Aggregate promoted and divide the results by 3 for promoted teams
    regr_model_single_new_team_results_agg = regr_model_new_team_results_agg.loc[
        regr_model_new_team_results_agg['team'] == 'Promoted']
    regr_model_single_new_team_results_agg = (
    regr_model_single_new_team_results_agg.iloc[:, 1:7].applymap(lambda x: x / 3))
    regr_model_single_new_team_results_agg.insert(0, 'team', 'Promoted')

    # , aggregate and add
    regr_model_this_seas_team_results_agg = pd.concat(
        [regr_model_this_seas_team_results_agg, regr_model_single_new_team_results_agg])
    regr_model_this_seas_team_results_agg = regr_model_this_seas_team_results_agg.reset_index(drop=True)

    # Finalise the model by creating the average number of games per season
    # No rounding at this stage, but accuracy should be ok
    number_seasons = len(regr_model_seasons)
    regr_model_team_names = pd.DataFrame(regr_model_this_seas_team_results_agg['team'])
    regr_model_this_seas_team_results_agg = (
    regr_model_this_seas_team_results_agg.iloc[:, 1:7].applymap(lambda x: x / number_seasons))
    regr_model_this_seas_team_results_agg = pd.concat([regr_model_team_names, regr_model_this_seas_team_results_agg],
                                                      axis=1)
    regr_model_this_seas_team_results_agg['season'] = regr_model_remain_seasons['season'].ix[i]

    # Add this season's values to main table
    regr_model_team_results_agg = pd.concat([regr_model_team_results_agg, regr_model_this_seas_team_results_agg])
    regr_model_team_results_agg = regr_model_team_results_agg.reset_index(drop=True)

    # Add to season list at bottom of loop
    regr_model_seasons = regr_model_seasons.append(regr_model_remain_seasons.ix[i])

print("Regression to mean")
print(regr_model_team_results_agg)



regr_model_team_results_agg.loc[regr_model_team_results_agg['team'] == 'Promoted']

# Build seprate dataframes for the home and away teams
regr_model_team_results_agg
regr_model_home_team_results_agg = regr_model_team_results_agg[['season', 'team', 'home_win', 'home_draw', 'home_lose']]
regr_model_home_team_results_agg.columns=['r_season', 'r_team', 'r_h_win', 'r_h_draw', 'r_h_lose']

regr_model_away_team_results_agg = regr_model_team_results_agg[['season', 'team', 'away_win', 'away_draw', 'away_lose']]
regr_model_away_team_results_agg.columns=['r_season', 'r_team', 'r_a_win', 'r_a_draw', 'r_a_lose']

'''
# Need to select Regression Figures for start of season into Features Table
predict_results = pd.merge(left=predict_results, right=home_results, how='left', left_on='home_team', right_on='team')
predict_results = predict_results.drop(['team'], axis=1)
predict_results = pd.merge(left=predict_results, right=away_results, how='left', left_on='away_team', right_on='team')
predict_results = predict_results.drop(['team'], axis=1)
'''
regr_model_home_team_results_agg

# Add regression season figures, then calculate, first home, then away
# Create a regression features table
regr_features = full_matches.copy()
set_promoted_teams(new_teams, regr_features)

# Ceate a df of unique teams
unique_teams = pd.DataFrame(full_matches.home_team.unique(), columns=['team'])

regr_home_features = pd.merge(left=regr_features, right=regr_model_home_team_results_agg, how='left',
                              left_on=['home_team', 'season'], right_on=['r_team', 'r_season'])
regr_home_features.sort_values(by=['home_team', 'date'], inplace=True)
regr_home_features.reset_index(drop=True, inplace=True)

# Set up home regression figures
previous_season = 'blank'
previous_team = 'blank'

for i, row in regr_home_features.iterrows():
    if (row['season'] != previous_season) or (row['home_team'] != previous_team):
        # New season or team
        previous_season = row['season']
        previous_team = row['home_team']
    else:
        # Wins
        if regr_home_features.ix[i - 1, 'home_win'] == 1:
            regr_home_features.ix[i, 'r_h_win'] = regr_home_features.ix[i - 1, 'r_h_win'] - 1
        else:
            regr_home_features.ix[i, 'r_h_win'] = regr_home_features.ix[i - 1, 'r_h_win']
        # Draws
        if regr_home_features.ix[i - 1, 'home_draw'] == 1:
            regr_home_features.ix[i, 'r_h_draw'] = regr_home_features.ix[i - 1, 'r_h_draw'] - 1
        else:
            regr_home_features.ix[i, 'r_h_draw'] = regr_home_features.ix[i - 1, 'r_h_draw']
        # Losses
        if regr_home_features.ix[i - 1, 'home_lose'] == 1:
            regr_home_features.ix[i, 'r_h_lose'] = regr_home_features.ix[i - 1, 'r_h_lose'] - 1
        else:
            regr_home_features.ix[i, 'r_h_lose'] = regr_home_features.ix[i - 1, 'r_h_lose']


def replace_negatives(value):
    if value < 0:
        return 0
    else:
        return value


regr_home_features['r_h_win'] = regr_home_features['r_h_win'].apply(replace_negatives)
regr_home_features['r_h_draw'] = regr_home_features['r_h_draw'].apply(replace_negatives)
regr_home_features['r_h_lose'] = regr_home_features['r_h_lose'].apply(replace_negatives)

print(regr_home_features)
# regr_features
# tmp_idx = (regr_model_home_team_results_agg['r_team'] == 'Arsenal') & (regr_model_home_team_results_agg['r_season'] == '2016/2017')


# Set up away regression figures
regr_away_features = pd.merge(left=regr_features, right=regr_model_away_team_results_agg, how='left', left_on=['away_team', 'season'], right_on=['r_team', 'r_season'])
regr_away_features.sort_values(by=['away_team', 'date'], inplace=True)
regr_away_features.reset_index(drop=True, inplace=True)
previous_season = 'blank'
previous_team = 'blank'

for i, row in regr_away_features.iterrows():
    if (row['season'] != previous_season) or (row['away_team'] != previous_team):
        # New season or team
        previous_season = row['season']
        previous_team = row['away_team']
    else:
        # Wins
        if regr_away_features.ix[i-1, 'away_win'] == 1:
            regr_away_features.ix[i, 'r_a_win'] = regr_away_features.ix[i-1, 'r_a_win'] - 1
        else:
            regr_away_features.ix[i, 'r_a_win'] = regr_away_features.ix[i-1, 'r_a_win']
        # Draws
        if regr_away_features.ix[i-1, 'away_draw'] == 1:
            regr_away_features.ix[i, 'r_a_draw'] = regr_away_features.ix[i-1, 'r_a_draw'] - 1
        else:
            regr_away_features.ix[i, 'r_a_draw'] = regr_away_features.ix[i-1, 'r_a_draw']
        # Losses
        if regr_away_features.ix[i-1, 'away_lose'] == 1:
            regr_away_features.ix[i, 'r_a_lose'] = regr_away_features.ix[i-1, 'r_a_lose'] - 1
        else:
            regr_away_features.ix[i, 'r_a_lose'] = regr_away_features.ix[i-1, 'r_a_lose']

regr_away_features['r_a_win'] = regr_away_features['r_a_win'].apply(replace_negatives)
regr_away_features['r_a_draw'] = regr_away_features['r_a_draw'].apply(replace_negatives)
regr_away_features['r_a_lose'] = regr_away_features['r_a_lose'].apply(replace_negatives)
print(regr_away_features)

# Convert to a percentage
regr_home_features['r_h_w_pct'] = regr_home_features['r_h_win'] / (regr_home_features['r_h_win'] +
                                                                   regr_home_features['r_h_draw'] +
                                                                   regr_home_features['r_h_lose'])
regr_home_features['r_h_d_pct'] = regr_home_features['r_h_draw'] / (regr_home_features['r_h_win'] +
                                                                   regr_home_features['r_h_draw'] +
                                                                   regr_home_features['r_h_lose'])

regr_home_features['r_h_l_pct'] = regr_home_features['r_h_lose'] / (regr_home_features['r_h_win'] +
                                                                   regr_home_features['r_h_draw'] +
                                                                   regr_home_features['r_h_lose'])

regr_home_features = regr_home_features.fillna(0)


# Convert away to a percentage
regr_away_features['r_a_w_pct'] = regr_away_features['r_a_win'] / (regr_away_features['r_a_win'] +
                                                                   regr_away_features['r_a_draw'] +
                                                                   regr_away_features['r_a_lose'])
regr_away_features['r_a_d_pct'] = regr_away_features['r_a_draw'] / (regr_away_features['r_a_win'] +
                                                                   regr_away_features['r_a_draw'] +
                                                                   regr_away_features['r_a_lose'])
regr_away_features['r_a_l_pct'] = regr_away_features['r_a_lose'] / (regr_away_features['r_a_win'] +
                                                                   regr_away_features['r_a_draw'] +
                                                                   regr_away_features['r_a_lose'])
regr_away_features = regr_away_features.fillna(0)

print("HOME")
print(regr_home_features)

print("AWAY")
print(regr_away_features)


#regr_away_features = regr_away_features.fillna(0)
print(regr_away_features[regr_away_features.isnull().any(axis=1)])

# Tidy up
final_regr_home_features = regr_home_features[['match_id', 'r_h_w_pct', 'r_h_d_pct', 'r_h_l_pct']]
final_regr_away_features = regr_away_features[['match_id', 'r_a_w_pct', 'r_a_d_pct', 'r_a_l_pct']]
final_regr_away_features.head()

print(final_regr_away_features[final_regr_away_features.isnull().any(axis=1)])
#regr_away_features.loc[regr_away_features['match_id'] == 4387]



# Build a separate features lookup df focused on each team. Can calculate various features:
# - Streak - Need to look at a team's run regardless of home or away
# - Goal Difference

# Loop through team list
team_features = pd.DataFrame()
for index, row in unique_teams.iterrows():
    # Then create a mask for that team - an index - on matches
    # Use that index to select all matches for a team
    single_team_matches_idx = (full_matches['home_team'] == row['team']) | (full_matches['away_team'] == row['team'])
    single_team_result = full_matches.loc[single_team_matches_idx]
    # , ['home_team', 'match_id', 'season', 'date',
    # 'home_team_goal', 'away_team_goal',
    # 'home_win', 'home_draw', 'home_lose',
    # 'away_win', 'away_draw', 'away_lose']]

    # Create the structure for streaks
    single_team_result['streak_team'] = row['team']
    team_features = pd.concat([team_features, single_team_result])

team_features.sort_values(by=['streak_team', 'date'], inplace=True)
team_features.reset_index(drop=True, inplace=True)

print("TEAM FEATURES")
print(team_features)


