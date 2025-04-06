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
        return self.used_teams[team] < current_date  # Can reuse team if it was used before current date
        
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
        
        # Pre-calculate team seeds and odds for faster lookup
        self.team_seeds = {}
        self.team_odds_cache = {}
        for _, row in team_odds.iterrows():
            team = row['teamName']
            self.team_seeds[team] = int(row['seed'])
            self.team_odds_cache[team] = {col: float(row[col]) for col in row.index if 'odds' in col}
        
        # If picks manager provided, validate picks
        if picks_manager:
            picks_manager.validate_picks(num_players)
            
            # Get all unique dates where any player has made a pick
            all_dates = set()
            for player_id in range(num_players):
                dates = picks_manager.get_dates_for_player(player_id)
                all_dates.update(dates)
            
            # Initialize players with their pre-specified picks
            for player_id, player in enumerate(self.players):
                # Get all dates for this player
                player_dates = set(picks_manager.get_dates_for_player(player_id))
                
                # Check if player is missing picks on any date where others made picks
                for date in sorted(all_dates):
                    try:
                        # Try both date formats
                        try:
                            date_obj = datetime.strptime(date, '%Y-%m-%d')
                        except ValueError:
                            date_obj = datetime.strptime(date, '%m/%d/%Y')
                        
                        # Convert to both formats
                        std_date = date_obj.strftime('%Y-%m-%d')
                        alt_date = date_obj.strftime('%m/%d/%Y')
                        
                        # Check if any other player has a pick for this date
                        other_players_have_picks = False
                        for other_id in range(num_players):
                            if other_id != player_id:
                                if picks_manager.has_pick(other_id, std_date) or picks_manager.has_pick(other_id, alt_date):
                                    other_players_have_picks = True
                                    break
                        
                        # If others have picks but this player doesn't, they're eliminated
                        if other_players_have_picks:
                            has_pick = picks_manager.has_pick(player_id, std_date) or picks_manager.has_pick(player_id, alt_date)
                            if not has_pick:
                                player.eliminated = True
                                player.eliminated_date = std_date
                                break
                        
                        # If player has a pick for this date and isn't eliminated, record it
                        if not player.eliminated:
                            pick = picks_manager.get_pick(player_id, std_date)
                            if not pick:
                                pick = picks_manager.get_pick(player_id, alt_date)
                            
                            if pick:  # Only record non-empty picks
                                # Normalize team name
                                pick = normalize_team_name(pick)
                                # Get team seed from cache
                                seed = self.team_seeds.get(pick, 16)  # Default to worst seed if not found
                                player.make_pick(std_date, pick, seed)
                                player.make_pick(alt_date, pick, seed)
                    except ValueError:
                        continue  # Skip invalid dates
    
    def calculate_pick_probabilities(self, available_teams: Set[str], round_name: str,
                                   player: Player) -> Dict[str, float]:
        """Calculate probability distribution for picking each available team."""
        probs = {}
        round_odds = f"{round_name}odds"
        round_num = int(round_name)
        
        # Get base probabilities from team odds cache
        total_odds = 0
        for team in available_teams:
            if not player.can_pick_team(team):
                continue
                
            try:
                # Handle Winner of games
                if team.startswith('Winner of:'):
                    # For Winner of games, use average odds of remaining teams
                    game_id = team.split(': ')[1]
                    game = self.schedule[self.schedule['gameID'] == game_id].iloc[0]
                    team_a = self.simulator.get_actual_teams(game['teamA'])
                    team_b = self.simulator.get_actual_teams(game['teamB'])
                    
                    if team_a and team_b:
                        odds_a = self.team_odds_cache.get(team_a, {}).get(round_odds, 0.5)
                        odds_b = self.team_odds_cache.get(team_b, {}).get(round_odds, 0.5)
                        odds = (odds_a + odds_b) / 2
                    else:
                        odds = 0.5
                else:
                    odds = self.team_odds_cache[team][round_odds]
                
                seed = self.team_seeds.get(team, 16)
                
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
            except (KeyError, ValueError):
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
        
        try:
            # Try both date formats
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                date_obj = datetime.strptime(date, '%m/%d/%Y')
            
            # Convert to both formats
            std_date = date_obj.strftime('%Y-%m-%d')
            alt_date = date_obj.strftime('%m/%d/%Y')
            
            for player in self.players:
                # Skip eliminated players
                if player.eliminated:
                    continue
                    
                # Skip players eliminated before this date
                if player.eliminated_date and player.eliminated_date <= std_date:
                    continue
                    
                # Check if player has a pre-specified pick for this date
                pick = None
                if self.picks_manager:
                    pick = self.picks_manager.get_pick(player.id, std_date)
                    if not pick:
                        pick = self.picks_manager.get_pick(player.id, alt_date)
                
                if pick:
                    # Handle Winner of games
                    if pick.startswith('Winner of:'):
                        game_id = pick.split(': ')[1]
                        if game_id in self.simulator.results:
                            pick = self.simulator.results[game_id].winner
                    pick = normalize_team_name(pick)
                    
                    # Get team seed from cache
                    seed = self.team_seeds.get(pick, 16)  # Default to worst seed if not found
                    player.make_pick(std_date, pick, seed)
                    player.make_pick(alt_date, pick, seed)  # Store pick in both formats
                    
                    # Record pick
                    if pick in team_picks:
                        team_picks[pick].append(player.id)
                    continue
                
                # Calculate pick probabilities for available teams
                probs = self.calculate_pick_probabilities(available_teams, round_name, player)
                if not probs:
                    # No valid teams to pick from, player is eliminated
                    player.eliminated = True
                    player.eliminated_date = std_date
                    continue
                
                # Make pick based on probabilities
                teams = list(probs.keys())
                probabilities = [probs[team] for team in teams]
                pick = np.random.choice(teams, p=probabilities)
                
                # Get team seed from cache
                seed = self.team_seeds.get(pick, 16)  # Default to worst seed if not found
                player.make_pick(std_date, pick, seed)
                player.make_pick(alt_date, pick, seed)  # Store pick in both formats
                
                # Record pick
                if pick in team_picks:
                    team_picks[pick].append(player.id)
                
        except Exception as e:
            print(f"Error simulating picks: {str(e)}")
            return team_picks
            
        return team_picks
    
    def process_results(self, date: str, winners: Set[str]) -> List[int]:
        """Process results for a given date and return list of surviving players."""
        if not winners:
            return []
            
        # Get normalized winner for this date
        winner = next(iter(winners))  # Take first winner since we only need one
        
        # Handle Winner of games
        if winner.startswith('Winner of:'):
            game_id = winner.split(': ')[1]
            game = self.schedule[self.schedule['gameID'] == game_id].iloc[0]
            winner = self.simulator.get_actual_teams(game['teamA'])
            if not winner:
                winner = self.simulator.get_actual_teams(game['teamB'])
            if not winner:
                print(f"Warning: No result found for previous game {game_id}")
                return []
        
        # Normalize winner team name
        winner = normalize_team_name(winner)
        print(f"\nNormalized winners for {date}: {winner}")
        
        # Check if any player has a pick for this date
        any_player_has_pick = False
        for player in self.players:
            if not player.eliminated and date in player.picks:
                any_player_has_pick = True
                break
        
        # Process each player's pick
        surviving_players = []
        for player in self.players:
            if player.eliminated:
                continue
                
            if date in player.picks:
                # Player made a pick
                if player.picks[date] == winner:
                    surviving_players.append(player.id)
                else:
                    player.eliminated = True
                    player.eliminated_date = date
            elif any_player_has_pick:
                # Player has no pick but others do - they're eliminated
                player.eliminated = True
                player.eliminated_date = date
            else:
                # No one has picks - everyone survives
                surviving_players.append(player.id)
        
        print(f"Surviving players after {date}: {surviving_players}")
        return surviving_players
    
    def determine_pool_winner(self, surviving_players: List[int]) -> int:
        """Determine the pool winner, using seed sum tiebreaker if necessary."""
        if not surviving_players:
            return None
            
        # Filter out players who were eliminated
        active_players = [pid for pid in surviving_players if not self.players[pid].eliminated]
        
        if not active_players:
            return None
        if len(active_players) == 1:
            return active_players[0]
            
        # Get players with minimum seed sum
        min_seeds = min(self.players[pid].seed_sum for pid in active_players)
        min_seed_players = [
            pid for pid in active_players
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