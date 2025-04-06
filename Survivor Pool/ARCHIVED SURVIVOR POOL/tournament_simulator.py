import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import random
from team_utils import normalize_team_name

@dataclass
class GameResult:
    winner: str
    loser: str
    game_id: str
    round_num: int
    winner_odds: float
    loser_odds: float

class Game:
    def __init__(self, team_a: str, team_b: str, round_num: int):
        """Initialize a game with two teams and round number."""
        self.team_a = team_a
        self.team_b = team_b
        self.round_num = round_num

class TournamentSimulator:
    def __init__(self, schedule_df, team_odds_df):
        self.schedule_df = schedule_df
        self.team_odds_df = team_odds_df
        self.results = {}  # game_id -> GameResult
        self.team_seeds = {}  # team_name -> seed
        
        # Initialize team seeds from the team odds DataFrame
        for _, row in team_odds_df.iterrows():
            if pd.notna(row['seed']):
                self.team_seeds[row['teamName']] = int(row['seed'])
    
    def get_team_seed(self, team_name):
        if team_name.startswith('Winner of:'):
            game_id = team_name.split('Winner of:')[1].strip()
            if game_id in self.results:
                return self.get_team_seed(self.results[game_id].winner)
        return self.team_seeds.get(team_name, None)
    
    def get_odds_column(self, round_num: int) -> str:
        """Get the odds column name for a given round number."""
        round_map = {
            64: '64odds',
            32: '32odds',
            16: '16odds',
            8: '8odds',
            4: '4odds',
            2: '2odds'
        }
        return round_map.get(round_num, '64odds')  # Default to 64odds if round not found
    
    def simulate_game(self, game_id: str, team1: str, team2: str, round_num: int) -> GameResult:
        """Simulate a single game between two teams."""
        odds_col = self.get_odds_column(round_num)
        
        # Get actual team names
        actual_team1 = self.get_actual_team(team1)
        actual_team2 = self.get_actual_team(team2)
        
        try:
            # Get team odds
            team1_odds = float(self.team_odds_df.loc[self.team_odds_df['teamName'] == actual_team1, odds_col].iloc[0])
            team2_odds = float(self.team_odds_df.loc[self.team_odds_df['teamName'] == actual_team2, odds_col].iloc[0])
            
            # If either team has 0 odds, they cannot win
            if team1_odds == 0:
                winner = team2
                loser = team1
                team1_prob = 0
            elif team2_odds == 0:
                winner = team1
                loser = team2
                team1_prob = 1
            else:
                # Get team seeds
                team1_seed = self.get_team_seed(team1)
                team2_seed = self.get_team_seed(team2)
                
                # Adjust odds based on seeds if available
                if team1_seed and team2_seed:
                    seed_factor = 0.05  # 5% adjustment per seed difference
                    seed_diff = team2_seed - team1_seed
                    team1_odds = min(0.95, max(0.05, team1_odds + (seed_diff * seed_factor)))
                    team2_odds = min(0.95, max(0.05, team2_odds - (seed_diff * seed_factor)))
                
                # Normalize probabilities
                total = team1_odds + team2_odds
                if total == 0:
                    # If both teams have 0 odds, use 0.5 for each
                    team1_prob = 0.5
                else:
                    team1_prob = team1_odds / total
                
                # Simulate the game
                winner = team1 if random.random() < team1_prob else team2
                loser = team2 if winner == team1 else team1
            
            # Store the result
            result = GameResult(winner=winner, loser=loser, game_id=game_id,
                              round_num=round_num, winner_odds=team1_odds if winner == team1 else team2_odds,
                              loser_odds=team2_odds if winner == team1 else team1_odds)
            self.results[game_id] = result
            
            # Update seeds for the winner
            winner_seed = self.get_team_seed(winner)
            if winner_seed:
                self.team_seeds[f'Winner of: {game_id}'] = winner_seed
            
            return result
        except Exception as e:
            print(f"Warning: Error processing team odds for game {game_id}: {str(e)}")
            print(f"Team1: {actual_team1}, Team2: {actual_team2}")
            print(f"Team1 odds: {team1_odds}, Team2 odds: {team2_odds}")
            raise
    
    def get_actual_team(self, team_placeholder):
        """Get the actual team name from a placeholder if needed."""
        if not isinstance(team_placeholder, str):
            print(f"Warning: team_placeholder is not a string: {team_placeholder}")
            return None
            
        if team_placeholder.startswith('Winner of:'):
            game_id = team_placeholder.split('Winner of:')[1].strip()
            if game_id in self.results:
                return self.get_actual_team(self.results[game_id].winner)
            print(f"Warning: No result found for game {game_id}")
            return None
            
        # Normalize team name
        normalized_team = normalize_team_name(team_placeholder)
        if normalized_team not in self.team_odds_df['teamName'].values:
            print(f"Warning: Normalized team name {normalized_team} not found in team_odds")
            return None
            
        return normalized_team
    
    def get_actual_teams(self, team_placeholder: str) -> str:
        """Convert 'Winner of: XYZ' placeholders to actual team names based on previous results."""
        if not isinstance(team_placeholder, str):
            return team_placeholder
            
        if not team_placeholder.startswith('Winner of:'):
            return normalize_team_name(team_placeholder)
            
        try:
            game_id = team_placeholder.split(': ')[1]
            if game_id not in self.results:
                # If we don't have results for this game yet, return the placeholder
                return team_placeholder
            return normalize_team_name(self.results[game_id].winner)
        except Exception as e:
            print(f"Warning: Error processing team placeholder '{team_placeholder}': {str(e)}")
            return team_placeholder
    
    def simulate_round(self, games):
        """Simulate all games in a round."""
        results = []
        for _, game in games.iterrows():
            team_a = game['teamA']
            team_b = game['teamB']
            
            result = self.simulate_game(game['gameID'], team_a, team_b, int(game['round']))
            results.append(result)
            
        return results
    
    def simulate_tournament(self):
        """Simulate the entire tournament."""
        self.results.clear()
        
        # Sort schedule by round (ascending) and day
        schedule = self.schedule_df.sort_values(['round', 'day'], ascending=[True, True])
        
        # Group games by day
        for day, day_games in schedule.groupby('day'):
            self.simulate_round(day_games)
    
    def get_teams_playing_on_date(self, target_date):
        """Get all teams playing on a specific date."""
        target_date = pd.to_datetime(target_date)
        day_games = self.schedule_df[self.schedule_df['day'] == target_date]
        
        teams = set()
        for _, game in day_games.iterrows():
            teams.add(self.get_actual_team(game['teamA']))
            teams.add(self.get_actual_team(game['teamB']))
        
        return teams
    
    def get_winners_on_date(self, target_date):
        """Get all teams that won on a specific date."""
        target_date = pd.to_datetime(target_date)
        day_games = self.schedule_df[self.schedule_df['day'] == target_date]
        
        winners = set()
        for _, game in day_games.iterrows():
            game_id = game['gameID']
            if game_id in self.results:
                winner = self.results[game_id].winner
                actual_winner = self.get_actual_team(winner)
                if actual_winner:
                    winners.add(actual_winner)
                else:
                    print(f"Warning: Could not get actual team for winner {winner} in game {game_id}")
        
        print(f"Winners on {target_date}: {winners}")
        return winners
    
    def get_games_for_date(self, date):
        """Get all games scheduled for a given date."""
        day_games = self.schedule_df[self.schedule_df['day'] == pd.to_datetime(date)]
        games = []
        for _, game in day_games.iterrows():
            games.append({
                'game_id': game['gameID'],
                'team1': self.get_actual_team(game['teamA']),
                'team2': self.get_actual_team(game['teamB']),
                'round': game['round']
            })
        return games 