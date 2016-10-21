import logging
from decimal import Decimal
from core.filehandling import get_or_create_match_record, \
    get_or_create_odds_record, \
    fetch_or_create_season_record, \
    fetch_or_create_team, \
    fetch_or_create_league
logger = logging.getLogger('django')
import csv

from datetime import datetime

with open('../gmblstatic/data/fixtures201617') as csvfile:
    data = list(tuple(rec) for rec in csv.reader(csvfile, delimiter=','))
from core.helperfunctions import convert_date_for_csv

if not data:
    successfully_added = 0
    return successfully_added
else:
    pass

successfully_added = 0
stats_index_list = []
odds_index_list = []

from core.models import League
from datetime import datetime
for row in data:
    home_team = Team.objects.filter(name=row[2])[0]
    away_team = Team.objects.filter(name=row[3])[0]
    league = League.objects.filter(name=row[0])[0]
    m, _ = Match.objects.get_or_create(
        match_date = datetime.strptime(row[1], "%d/%m/%y %H:%M").date(),
        home_team = home_team,
        away_team = away_team,
        league = league
    )
    if _:
        pass
    else:
        m.save()

bulk_list = []
for row in data:
    home_team = Team.objects.filter(name=row[2])[0]
    away_team = Team.objects.filter(name=row[3])[0]
    league = League.objects.filter(name=row[0])[0]
    m = Match(home_team=home_team, away_team=away_team, match_date=datetime.strptime(row[1], "%d/%m/%y %H:%M").date(), league=league)
    bulk_list.append(m)

for row in data:
    # Sets a few conditions that determine the condition of the data we're about to parse
    # as it varies a lot across the ~300 leagues we have data for
    stats_list = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
    odds_list = ["B365H", "B365D", "B365A"]
    try:
        if not row[0]:
            logger.debug('Skipping row, empty')
        elif row[0] == "Div":
            logger.debug("Row is %s", str(row))
            for stat in stats_list:
                if stat in row:
                    stats_index_list.append(row.index(stat))
                else:
                    stats_index_list.append(None)
            for odds in odds_list:
                if odds in row:
                    odds_index_list.append(row.index(odds))
                else:
                    stats_index_list.append(None)

            logger.debug("Stats and index lists are here:")
            logger.debug(stats_list)
            logger.debug(stats_index_list)
        else:
            # try:
            logger.debug(row)
            logger.debug("type is %s", type(row[1]))
            if not isinstance(row[1], datetime):
                working_date = convert_date_for_csv(row[1])
            else:
                pass
            home_team = fetch_or_create_team(row[2])
            away_team = fetch_or_create_team(row[3])
            league = fetch_or_create_league(row[0])
            home_season = fetch_or_create_season_record(home_team, league, working_date)
            away_season = fetch_or_create_season_record(away_team, league, working_date)

            # except(ValueError), e:
            # logger.error('Something went wrong in converting the dates for season data', row[0], row[1], e)

            stats_slice = []
            if stats_index_list:
                for position in stats_index_list:
                    try:
                        # logger.debug("Adding row index to index list: %s", str(row[position]))
                        if row[position] == '':
                            stats_slice.append(None)
                        else:
                            stats_slice.append(int(Decimal(row[position])))
                    except (TypeError), e:
                        stats_slice.append(None)
                        logger.debug("TypeError %s, appending 1xNone ", e)
                print
                stats_slice
            else:
                stats_slice = [None, None, None, None, None, None, None, None, None, None]
                logger.debug("No stats_index_list so have appended 10xNone")
            logger.debug("Stats slice goes here, is %d long, looks like this: %s", len(stats_slice),
                         str(stats_slice))

            odds_slice = []
            if odds_index_list:
                for position in odds_index_list:
                    try:
                        # logger.debug("Adding row index to odds index list: %s", str(row[position]))
                        if row[position] == '':
                            odds_slice.append(None)
                        else:
                            odds_slice.append(row[position])
                    except (TypeError), e:
                        odds_slice.append(None)
                        logger.debug("TypeError %s, appending 1xNone in odds ", e)
            else:
                odds_slice = [None, None, None, None, None, None, None, None, None, None]
                # logger.debug("No stats_index_list so have appended 10xNone")
                # logger.debug("Odds slice goes here, is %d long, looks like this: %s" % (len(odds_slice), str(odds_slice)))

            if row[4] is not '':
                goals_slice = row[4:7]
            else:
                goals_slice = [None, None, None, None]

            print "under goals slice", goals_slice
            try:
                print "In match creation"
                match, _ = get_or_create_match_record(league, home_team, away_team, working_date, goals_slice,
                                                      stats_slice)
            # This is all horrible, redundant code that needs refactored.
            except(ValueError), e:
                print "In value error %s", e
                if stats_slice[0] is not None:
                    for i, number in enumerate(stats_slice):
                        print
                        type, number
                        temp = str(number)
                        stats_slice[i] = int(temp)
                    for i, number in enumerate(goals_slice):
                        goals_slice[i] = round(number)
                    match, _ = get_or_create_match_record(league, home_team, away_team, working_date, goals_slice,
                                                          stats_slice)
                else:
                    match, _ = get_or_create_match_record(league, home_team, away_team, working_date, goals_slice,
                                                          stats_slice)
            if odds_slice[0] is not None:
                get_or_create_odds_record(match, "home", odds_slice[0])
                get_or_create_odds_record(match, "draw", odds_slice[1])
                get_or_create_odds_record(match, "away", odds_slice[2])
            else:
                pass
            successfully_added += 1
    except(IndexError), e:
        logger.error(e)
        continue