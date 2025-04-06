import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import random
from player_picks import PlayerPicksManager
from tournament_simulator import TournamentSimulator
from datetime import datetime
from team_utils import normalize_team_name

class Player:
    def __init__(self, id: int, favorites_factor: float, variance_factor: float):
        self.id = id
        self.favorites_factor = favorites_factor
        self.variance_factor = variance_factor
        self.picks: Dict[str, str] = {}  # date -> team
        self.eliminated = False
        self.eliminated_date = None
        self.used_teams: Dict[str, str] = {}  # team -> date used
        self.seed_sum = 0
        
    def can_pick_team(self, team: str, current_date: str = None) -> bool:
        """Check if a team can be picked by this player on the given date."""
        if team not in self.used_teams:
            return True
        if current_date is None:
            return False
        return self.used_teams[team] >= current_date  # Can reuse team if it was used on or after current date
        
    def make_pick(self, date: str, team: str, seed: int):
        """Record a pick for this player."""
        self.picks[date] = team
        if team not in self.used_teams or date < self.used_teams[team]:
            self.used_teams[team] = date
            self.seed_sum += seed

class PoolSimulator:
    def __init__(self, num_players: int, favorites_factor: int, variance_factor: int,
                 schedule: pd.DataFrame, team_odds: pd.DataFrame, picks_manager: PlayerPicksManager = None):
        """Initialize the pool simulator."""
        self.num_players = num_players
        self.favorites_factor = favorites_factor
        self.variance_factor = variance_factor
        self.schedule = schedule
        self.team_odds = team_odds
        self.picks_manager = picks_manager
        self.players = [Player(i, favorites_factor / 10.0, variance_factor / 10.0) for i in range(num_players)]
        self.simulator = TournamentSimulator(schedule, team_odds)
        
        # If picks manager provided, validate picks
        if picks_manager:
            picks_manager.validate_picks(num_players)
            
            # Initialize players with their pre-specified picks
            for player_id, player in enumerate(self.players):
                # Get all dates for this player
                dates = picks_manager.get_dates_for_player(player_id)
                for date in dates:
                    pick = picks_manager.get_pick(player_id, date)
                    if pick:  # Only record non-empty picks
                        # Get team seed
                        try:
                            seed = int(self.team_odds[self.team_odds['teamName'] == pick]['seed'].iloc[0])
                            player.make_pick(date, pick, seed)
                        except (IndexError, ValueError):
                            print(f"WARNING: Could not find seed for team {pick}")
                            player.make_pick(date, pick, 16)  # Default to worst seed
        
    def calculate_pick_probabilities(self, available_teams: Set[str], round_name: str,
                                   player: Player) -> Dict[str, float]:
        """Calculate probability distribution for picking each available team."""
        probs = {}
        round_odds = f"{round_name}odds"
        round_num = int(round_name)
        
        # Get base probabilities from team odds
        total_odds = 0
        for team in available_teams:
            if not player.can_pick_team(team):
                continue
                
            try:
                odds = float(self.team_odds[self.team_odds['teamName'] == team][round_odds].iloc[0])
                seed = int(self.team_odds[self.team_odds['teamName'] == team]['seed'].iloc[0])
                
                # Apply favorites factor with seed consideration
                seed_factor = 1.0 + (16 - seed) / 32  # Higher seeds get slight boost
                odds = max(0.0001, odds ** (1 / player.favorites_factor) * seed_factor)
                
                # Apply round-specific adjustments
                if round_num <= 32:
                    # Early rounds favor favorites more
                    odds = odds ** (1 / (1 + player.favorites_factor/10))
                elif round_num >= 4:
                    # Later rounds favor underdogs more
                    odds = odds ** (1 + player.favorites_factor/10)
                
                probs[team] = odds
                total_odds += odds
            except (IndexError, ValueError):
                # If team not found or invalid odds, assign small probability
                probs[team] = 0.0001
                total_odds += 0.0001
            
        # If no valid teams, return empty dict
        if not probs:
            return {}
            
        # Normalize probabilities
        for team in probs:
            probs[team] /= total_odds
            
        # Apply variance factor with improved distribution
        if player.variance_factor < 1:
            # Lower variance - weighted toward favorites
            max_prob = max(probs.values())
            for team in probs:
                probs[team] = (probs[team] + (1 - player.variance_factor) * max_prob) / (2 - player.variance_factor)
        else:
            # Higher variance - weighted toward underdogs
            min_prob = min(probs.values())
            for team in probs:
                probs[team] = (probs[team] + (player.variance_factor - 1) * min_prob) / player.variance_factor
        
        # Ensure probabilities sum to 1
        total_prob = sum(probs.values())
        if abs(total_prob - 1.0) > 1e-10:  # Allow for small floating point errors
            for team in probs:
                probs[team] /= total_prob
                    
        return probs
    
    def simulate_player_picks(self, date: str, available_teams: Set[str],
                            round_name: str) -> Dict[str, List[int]]:
        """Simulate picks for all active players on a given date."""
        team_picks: Dict[str, List[int]] = {team: [] for team in available_teams}
        
        for player in self.players:
            if player.eliminated:
                continue
                
            # Check if player has a pre-specified pick for this date
            if self.picks_manager and self.picks_manager.has_pick(player.id, date):
                pick = self.picks_manager.get_pick(player.id, date)
                if pick not in available_teams:
                    # Pre-specified pick is invalid (team not playing), player is eliminated
                    player.eliminated = True
                    player.eliminated_date = date
                    continue
                    
                if not player.can_pick_team(pick, date):
                    # Pre-specified pick is invalid (team already used), player is eliminated
                    player.eliminated = True
                    player.eliminated_date = date
                    continue
                    
                # Get team seed
                try:
                    seed = int(self.team_odds[self.team_odds['teamName'] == pick]['seed'].iloc[0])
                    player.make_pick(date, pick, seed)
                except (IndexError, ValueError):
                    print(f"WARNING: Could not find seed for team {pick}")
                    player.make_pick(date, pick, 16)  # Default to worst seed
                    
                team_picks[pick].append(player.id)
                continue
                
            # Calculate pick probabilities for simulated pick
            probs = self.calculate_pick_probabilities(available_teams, round_name, player)
            
            if not probs:
                # No valid teams to pick - player is eliminated
                player.eliminated = True
                player.eliminated_date = date
                continue
                
            # Make pick
            teams = list(probs.keys())
            probabilities = [probs[team] for team in teams]
            
            # Ensure probabilities sum to 1
            total_prob = sum(probabilities)
            if abs(total_prob - 1.0) > 1e-10:  # Allow for small floating point errors
                probabilities = [p / total_prob for p in probabilities]
                
            pick = np.random.choice(teams, p=probabilities)
            
            # Get team seed
            try:
                seed = int(self.team_odds[self.team_odds['teamName'] == pick]['seed'].iloc[0])
                player.make_pick(date, pick, seed)
            except (IndexError, ValueError):
                print(f"WARNING: Could not find seed for team {pick}")
                player.make_pick(date, pick, 16)  # Default to worst seed
                
            team_picks[pick].append(player.id)
            
        return team_picks
    
    def process_results(self, date: datetime, winners: Set[str]) -> List[int]:
        """Process game results for a date, return list of surviving player IDs."""
        surviving_players = []
        normalized_winners = {normalize_team_name(w) for w in winners}
        
        for player in self.players:
            # Skip already eliminated players
            if player.eliminated:
                continue
                
            # Get player's pick for this date
            date_str = date.strftime('%Y-%m-%d')
            pick = player.picks.get(date_str)
            
            # Only eliminate if player has a pick for this date and it loses
            # AND there are winners for this date (meaning games have been played)
            if pick and normalized_winners:
                pick = normalize_team_name(pick)
                # If player's pick is not in winners, they lost
                if pick not in normalized_winners:
                    player.eliminated = True
                    player.eliminated_date = date
                    continue
            
            # Player survived this date (either had a winning pick, no pick, or no games played)
            surviving_players.append(player.id)
        
        return surviving_players
    
    def determine_pool_winner(self, surviving_players: List[int]) -> int:
        """Determine the pool winner, using seed sum tiebreaker if necessary."""
        if not surviving_players:
            return None
        if len(surviving_players) == 1:
            return surviving_players[0]
            
        # Get players with minimum seed sum
        min_seeds = min(self.players[pid].seed_sum for pid in surviving_players)
        min_seed_players = [
            pid for pid in surviving_players
            if self.players[pid].seed_sum == min_seeds
        ]
        
        if len(min_seed_players) == 1:
            return min_seed_players[0]
            
        # If still tied, use number of picks as secondary tiebreaker
        pick_counts = {
            pid: len(self.players[pid].picks)
            for pid in min_seed_players
        }
        min_picks = min(pick_counts.values())
        
        min_pick_players = [
            pid for pid in min_seed_players
            if pick_counts[pid] == min_picks
        ]
        
        # If still tied, use coin flip
        return random.choice(min_pick_players) 