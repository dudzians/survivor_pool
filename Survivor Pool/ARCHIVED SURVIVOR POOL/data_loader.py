import pandas as pd
from datetime import datetime
import numpy as np
from team_utils import normalize_team_name, validate_team_names
import logging

def load_schedule(file_path: str) -> pd.DataFrame:
    """Load schedule from CSV file."""
    logging.info(f"Loading schedule from {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['day'] = pd.to_datetime(df['day'])
    
    # Clean and normalize team names
    df['teamA'] = df['teamA'].str.strip()
    df['teamB'] = df['teamB'].str.strip()
    df['teamA'] = df['teamA'].apply(normalize_team_name)
    df['teamB'] = df['teamB'].apply(normalize_team_name)
    
    # Validate required columns
    required_columns = ['day', 'round', 'teamA', 'teamB']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Final validation
    logging.info(f"Games loaded: {len(df)}")
    
    return df

def load_team_odds(file_path: str) -> pd.DataFrame:
    """Load team odds from CSV file."""
    logging.info(f"Loading odds from {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Clean and normalize team names
    df['teamName'] = df['teamName'].str.strip()
    df['teamName'] = df['teamName'].apply(normalize_team_name)
    
    # Validate required columns
    required_columns = ['seed', 'region', 'teamName', 'conf', 'intourney']
    odds_columns = ['64odds', '32odds', '16odds', '8odds', '4odds', '2odds']
    
    missing_columns = [col for col in required_columns + odds_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Process odds columns
    for col in odds_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
            
    # Fill missing odds with 0
    for col in odds_columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            logging.info(f"Filling {missing_count} missing values in {col}")
            df[col] = df[col].fillna(0)
    
    # Final validation
    logging.info(f"\nTeams loaded: {len(df)}")
    
    return df

def get_games_for_date(schedule: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """Get all games scheduled for a specific date."""
    try:
        target_date = pd.to_datetime(target_date)
        games = schedule[schedule['day'] == target_date]
        
        if games.empty:
            print(f"WARNING: No games found for date {target_date}")
        
        return games
    except Exception as e:
        raise ValueError(f"Error processing date {target_date}: {str(e)}")

def get_teams_playing_on_date(schedule: pd.DataFrame, target_date: str) -> set:
    """Get all teams playing on a specific date."""
    games = get_games_for_date(schedule, target_date)
    teams = set()
    
    for _, game in games.iterrows():
        if not isinstance(game['teamA'], str) or not game['teamA'].startswith('Winner'):
            teams.add(normalize_team_name(game['teamA'].strip()))
        if not isinstance(game['teamB'], str) or not game['teamB'].startswith('Winner'):
            teams.add(normalize_team_name(game['teamB'].strip()))
    
    return teams

def validate_data(schedule: pd.DataFrame, team_odds: pd.DataFrame) -> None:
    """Validate that all teams in schedule are in team_odds."""
    # Get all actual teams from schedule (excluding "Winner of:" teams)
    schedule_teams = set()
    for _, game in schedule.iterrows():
        for team in [game['teamA'], game['teamB']]:
            if isinstance(team, str) and not team.startswith('Winner of:'):
                schedule_teams.add(team)
    
    odds_teams = set(team_odds['teamName'].unique())
    
    missing_teams = schedule_teams - odds_teams
    if missing_teams:
        raise ValueError(f"Teams in schedule but not in odds: {missing_teams}")
    
    logging.info(f"Data validation complete")

    # Get all teams from schedule
    schedule_teams = set()
    for _, game in schedule.iterrows():
        if not isinstance(game['teamA'], str) or not game['teamA'].startswith('Winner'):
            schedule_teams.add(game['teamA'])
        if not isinstance(game['teamB'], str) or not game['teamB'].startswith('Winner'):
            schedule_teams.add(game['teamB'])
    
    # Get all teams from odds
    odds_teams = set(team_odds['teamName'])
    
    # Validate team names
    missing_teams, potential_mismatches = validate_team_names(schedule_teams, odds_teams)
    
    if missing_teams:
        print(f"WARNING: Teams in schedule but not in odds: {missing_teams}")
    if potential_mismatches:
        print(f"WARNING: Teams in odds but not in schedule: {potential_mismatches}")
    
    # Validate seeds
    if not all(1 <= seed <= 16 for seed in team_odds['seed']):
        print("WARNING: Invalid seeds found in teams data")
    
    # Validate tournament structure
    rounds = ['64odds', '32odds', '16odds', '8odds', '4odds', '2odds']
    for i in range(1, len(rounds)):
        if not all(team_odds[rounds[i]] <= team_odds[rounds[i-1]]):
            print(f"WARNING: Found teams with increasing odds between {rounds[i-1]} and {rounds[i]}") 