import pandas as pd
from typing import Dict, Set, Optional, List
from datetime import datetime
from team_utils import normalize_team_name

class PlayerPicksManager:
    def __init__(self, picks_file: Optional[str] = None):
        """
        Initialize with optional picks_file. If not provided, all picks will be simulated.
        Expected CSV format:
        player_id,2025-03-20,2025-03-21,2025-03-22,...
        0,Houston,,Tennessee,...
        1,,Duke,,...
        """
        self.picks: Dict[int, Dict[str, str]] = {}  # player_id -> {date -> team}
        if picks_file:
            print(f"Loading picks from {picks_file}")
            self._load_picks(picks_file)
    
    def _load_picks(self, picks_file: str) -> None:
        """Load picks from CSV file."""
        # Read CSV file
        df = pd.read_csv(picks_file)
        
        # First column should be player_id, rest should be dates
        date_columns = [col for col in df.columns if col != 'player_id']
        
        # Convert date columns from M/D/YYYY to YYYY-MM-DD
        date_mapping = {}  # Store mapping of old to new column names
        for date_col in date_columns:
            try:
                # Parse date in M/D/YYYY format
                date = datetime.strptime(date_col, '%m/%d/%Y')
                # Convert to YYYY-MM-DD format
                new_col = date.strftime('%Y-%m-%d')
                # Store mapping
                date_mapping[date_col] = new_col
                # Rename column
                df = df.rename(columns={date_col: new_col})
            except ValueError as e:
                print(f"Error parsing date column {date_col}: {str(e)}")
                # Try alternate format
                try:
                    date = pd.to_datetime(date_col).strftime('%Y-%m-%d')
                    date_mapping[date_col] = date
                    df = df.rename(columns={date_col: date})
                except:
                    raise ValueError(f"Invalid date column format: {date_col}. Expected M/D/YYYY")
        
        # Process each player's picks
        for _, row in df.iterrows():
            player_id = int(row['player_id'])
            self.picks[player_id] = {}
            
            for old_date, new_date in date_mapping.items():
                if pd.notna(row[new_date]) and str(row[new_date]).strip():  # Check if pick exists and is not empty
                    team = str(row[new_date]).strip()
                    # Normalize team name
                    normalized_team = normalize_team_name(team)
                    self.picks[player_id][new_date] = normalized_team
    
    def get_pick(self, player_id: int, date: str) -> Optional[str]:
        """Get a player's pick for a specific date if it exists."""
        return self.picks.get(player_id, {}).get(date)
    
    def has_pick(self, player_id: int, date: str) -> bool:
        """Check if a player has a pre-specified pick for a date."""
        return bool(self.get_pick(player_id, date))
    
    def get_dates_for_player(self, player_id: int) -> List[str]:
        """Get all dates for which a player has picks."""
        return sorted(list(self.picks.get(player_id, {}).keys()))
    
    def get_used_teams(self, player_id: int, before_date: str) -> Set[str]:
        """Get all teams used by a player before a specific date."""
        used_teams = set()
        target_date = datetime.strptime(before_date, '%Y-%m-%d')
        
        player_picks = self.picks.get(player_id, {})
        for date, team in player_picks.items():
            if datetime.strptime(date, '%Y-%m-%d') < target_date:
                used_teams.add(team)
        
        return used_teams
    
    def validate_picks(self, num_players: int) -> None:
        """
        Validate that the picks file is properly formatted.
        """
        if not self.picks:
            return
            
        # Check no duplicate teams per player
        for player_id, picks in self.picks.items():
            used_teams = set()
            for date, team in sorted(picks.items()):  # Sort by date to check in chronological order
                if team in used_teams:
                    raise ValueError(f"Player {player_id} picked {team} multiple times")
                used_teams.add(team) 