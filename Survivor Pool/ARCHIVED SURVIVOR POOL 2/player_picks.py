import pandas as pd
from typing import Dict, Set, Optional, List
from datetime import datetime
from team_utils import normalize_team_name

class PlayerPicksManager:
    def __init__(self, picks_file: Optional[str] = None):
        """
        Initialize with optional picks_file. If not provided, all picks will be simulated.
        Expected CSV format:
        player_id,3/20/2025,3/21/2025,3/22/2025,...
        0,Houston,,Tennessee,...
        1,,Duke,,...
        """
        self.picks: Dict[int, Dict[str, str]] = {}  # player_id -> {date -> team}
        self.num_players = 0  # Initialize num_players
        if picks_file:
            print(f"Loading picks from {picks_file}")
            self._load_picks(picks_file)
    
    def _load_picks(self, picks_file: str) -> None:
        """Load picks from CSV file."""
        try:
            print("Reading CSV file...")
            # Read CSV file
            df = pd.read_csv(picks_file)
            self.picks = {}
            self.num_players = len(df)  # Set num_players based on number of rows
            
            # Convert date columns to datetime and rename to YYYY-MM-DD format
            date_columns = [col for col in df.columns if col != 'player_id']
            date_mapping = {}
            
            for date_col in date_columns:
                try:
                    # Try M/D/YYYY format first
                    date = datetime.strptime(date_col, '%m/%d/%Y')
                    new_col = date.strftime('%Y-%m-%d')
                    date_mapping[date_col] = new_col
                except ValueError:
                    try:
                        # Try YYYY-MM-DD format
                        datetime.strptime(date_col, '%Y-%m-%d')
                        new_col = date_col
                        date_mapping[date_col] = new_col
                    except ValueError:
                        raise ValueError(f"Invalid date column format: {date_col}. Expected M/D/YYYY or YYYY-MM-DD")
            
            # Store picks for each player
            for _, row in df.iterrows():
                player_id = int(row['player_id'])
                if player_id not in self.picks:
                    self.picks[player_id] = {}
                
                # Process each date column
                for old_date, new_date in date_mapping.items():
                    pick = row[old_date]
                    if pd.notna(pick) and pick.strip():  # Only store non-empty picks
                        # Normalize team name to match teams.csv format
                        normalized_pick = normalize_team_name(pick.strip())
                        self.picks[player_id][new_date] = normalized_pick
                        # Also store with M/D/YYYY format for compatibility
                        alt_date = datetime.strptime(new_date, '%Y-%m-%d').strftime('%m/%d/%Y')
                        self.picks[player_id][alt_date] = normalized_pick
                        
            print(f"Loaded picks for {len(self.picks)} players")
            
        except Exception as e:
            raise ValueError(f"Error loading picks file: {str(e)}")
    
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
            try:
                # Try both date formats
                try:
                    pick_date = datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    pick_date = datetime.strptime(date, '%m/%d/%Y')
                
                if pick_date < target_date:
                    used_teams.add(team)
            except ValueError:
                continue
        
        return used_teams
    
    def validate_picks(self, num_players: int) -> None:
        """
        Validate that the picks file is properly formatted.
        """
        if not self.picks:
            return
            
        # Check that all player IDs are valid
        for player_id in self.picks:
            if not isinstance(player_id, int) or player_id < 0 or player_id >= num_players:
                raise ValueError(f"Invalid player ID found in picks file: {player_id}")
            
        # Check no duplicate teams per player
        for player_id, picks in self.picks.items():
            used_teams = set()
            # Get unique dates (ignoring format)
            unique_dates = set()
            for date in sorted(picks.keys()):
                try:
                    # Try both date formats
                    try:
                        date_obj = datetime.strptime(date, '%Y-%m-%d')
                    except ValueError:
                        date_obj = datetime.strptime(date, '%m/%d/%Y')
                    
                    if date_obj not in unique_dates:
                        unique_dates.add(date_obj)
                        team = picks[date]
                        if team in used_teams:
                            raise ValueError(f"Player {player_id} picked {team} multiple times")
                        used_teams.add(team)
                except ValueError:
                    continue 