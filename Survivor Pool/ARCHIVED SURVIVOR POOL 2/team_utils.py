import re

# Comprehensive mapping of team name variations
TEAM_NAME_MAPPING = {
    'Michigan State': 'Michigan St.',
    'Michigan St': 'Michigan St.',
    'MSU': 'Michigan St.',
    'Iowa State': 'Iowa St.',
    'Iowa St': 'Iowa St.',
    'ISU': 'Iowa St.',
    'Colorado State': 'Colorado St.',
    'Colorado St': 'Colorado St.',
    'CSU': 'Colorado St.',
    'McNeese State': 'McNeese St.',
    'McNeese St': 'McNeese St.',
    'Mississippi State': 'Mississippi St.',
    'Mississippi St': 'Mississippi St.',
    'MSST': 'Mississippi St.',
    'Alabama State': 'Alabama St.',
    'Alabama St': 'Alabama St.',
    'ALST': 'Alabama St.',
    'Norfolk State': 'Norfolk St.',
    'Norfolk St': 'Norfolk St.',
    'NFST': 'Norfolk St.',
    'Mount Saint Mary\'s': 'Mount St. Mary\'s',
    'Mount St Mary\'s': 'Mount St. Mary\'s',
    'MTST': 'Mount St. Mary\'s',
    'Saint Mary\'s': "Saint Mary's",
    'St Mary\'s': "Saint Mary's",
    'STMY': "Saint Mary's",
    'Saint John\'s': "St. John's",
    'St John\'s': "St. John's",
    'STJN': "St. John's",
    'UC San Diego': 'UC San Diego',
    'UCSD': 'UC San Diego',
    'UNC Wilmington': 'UNC Wilmington',
    'UNCW': 'UNC Wilmington',
    'SIU Edwardsville': 'SIU Edwardsville',
    'SIUE': 'SIU Edwardsville',
    'Grand Canyon': 'Grand Canyon',
    'GCU': 'Grand Canyon',
    'High Point': 'High Point',
    'Robert Morris': 'Robert Morris',
    'Nebraska Omaha': 'Nebraska Omaha',
    'Liberty': 'Liberty',
    'Akron': 'Akron',
    'Bryant': 'Bryant',
    'Lipscomb': 'Lipscomb',
    'Montana': 'Montana',
    'Vanderbilt': 'Vanderbilt',
    'Oklahoma': 'Oklahoma',
    'Georgia': 'Georgia',
    'Troy': 'Troy',
    'Xavier': 'Xavier',
    'Wofford': 'Wofford',
    'Yale': 'Yale',
    'UNLV': 'UNLV',
    'UCLA': 'UCLA',
    'USC': 'USC',
    'BYU': 'BYU',
    'UConn': 'Connecticut',
    'Connecticut': 'Connecticut',
    'UNC': 'North Carolina',
    'North Carolina': 'North Carolina',
    'Duke': 'Duke',
    'UK': 'Kentucky',
    'Kentucky': 'Kentucky',
    'KU': 'Kansas',
    'Kansas': 'Kansas',
}

def normalize_team_name(team_name: str) -> str:
    """
    Normalize team names to match the format in teams.csv.
    Returns the normalized team name.
    """
    if not isinstance(team_name, str):
        return team_name
        
    # Remove any leading/trailing whitespace
    team_name = team_name.strip()
    
    # Check for direct mapping first
    if team_name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[team_name]
    
    # Try to match against known variations
    for old, new in TEAM_NAME_MAPPING.items():
        if old in team_name:
            return new
    
    # Handle abbreviations
    if len(team_name) <= 4 and team_name.isupper():
        for old, new in TEAM_NAME_MAPPING.items():
            if old.startswith(team_name):
                return new
    
    return team_name

def validate_team_names(schedule_teams: set, odds_teams: set) -> tuple[set, set]:
    """
    Validate team names between schedule and odds files.
    Returns sets of missing teams and potential mismatches.
    """
    schedule_teams = {normalize_team_name(team) for team in schedule_teams}
    odds_teams = {normalize_team_name(team) for team in odds_teams}
    
    missing_teams = schedule_teams - odds_teams
    potential_mismatches = odds_teams - schedule_teams
    
    return missing_teams, potential_mismatches 