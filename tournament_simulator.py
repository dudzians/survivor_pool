import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import random
import csv
import multiprocessing
from functools import partial

# Add a global debug flag that will be set via command line
DEBUG = False

def load_seed_boosts(path='seed_boosts.csv'):
    boosts = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            round_label = row['round']
            seed = int(row['seed'])
            boost = float(row['boost'])
            if round_label not in boosts:
                boosts[round_label] = {}
            boosts[round_label][seed] = boost
    return boosts

SEED_BOOSTS = load_seed_boosts()

def load_data():
    # Load data from CSV files
    schedule_df = pd.read_csv('schedule.csv')
    teams_df = pd.read_csv('teams.csv')
    picks_df = pd.read_csv('sample_picks.csv', index_col='player_id')
    
    # Create team mapping
    team_mapping = create_team_mapping(teams_df, picks_df, schedule_df)
    
    return schedule_df, teams_df, picks_df, team_mapping

def create_team_mapping(teams_df, picks_df, schedule_df):
    """Create a mapping of team names to their official names."""
    # Get all unique team names from all sources
    teams_from_teams = set(teams_df['teamName'])
    teams_from_picks = set(picks_df.iloc[:, 1:].stack().unique())
    teams_from_schedule = set(schedule_df['teamA'].str.strip().unique()) | set(schedule_df['teamB'].str.strip().unique())
    
    # Create mapping dictionary
    mapping = {}
    
    # First, map teams from teams.csv to themselves
    for team in teams_from_teams:
        mapping[team] = team
    
    # Then map variations from picks and schedule to official names
    for team in teams_from_picks | teams_from_schedule:
        if team not in mapping:
            # Try to find a match in teams.csv
            for official_team in teams_from_teams:
                if team.strip().lower() in official_team.lower() or official_team.lower() in team.strip().lower():
                    mapping[team] = official_team
                    break
            else:
                # If no match found, use the original name
                mapping[team] = team.strip()
    
    # Add specific mappings for teams with different names
    mapping['Colorado State'] = 'Colorado St.'
    mapping['Iowa State'] = 'Iowa St.'
    mapping['McNeese State'] = 'McNeese St.'
    mapping['Michigan State'] = 'Michigan St.'
    mapping['Mississippi State'] = 'Mississippi St.'
    mapping['UC San Diego'] = 'UC San Diego'
    mapping['UNC Wilmington'] = 'UNC Wilmington'
    mapping['SIU Edwardsville'] = 'SIU Edwardsville'
    mapping['Mount St. Mary\'s'] = 'Mount St. Mary\'s'
    mapping['Robert Morris'] = 'Robert Morris'
    mapping['Norfolk St.'] = 'Norfolk St.'
    mapping['Nebraska Omaha'] = 'Nebraska Omaha'
    mapping['Grand Canyon'] = 'Grand Canyon'
    mapping['High Point'] = 'High Point'
    mapping['Liberty'] = 'Liberty'
    mapping['Lipscomb'] = 'Lipscomb'
    mapping['Louisville'] = 'Louisville'
    mapping['Marquette'] = 'Marquette'
    mapping['Memphis'] = 'Memphis'
    mapping['Mississippi'] = 'Mississippi'
    mapping['Oregon'] = 'Oregon'
    mapping['Saint Mary\'s'] = 'Saint Mary\'s'
    mapping['St. John\'s'] = 'St. John\'s'
    mapping['Texas A&M'] = 'Texas A&M'
    mapping['UCLA'] = 'UCLA'
    mapping['Wisconsin'] = 'Wisconsin'
    mapping['Akron'] = 'Akron'
    mapping['Alabama St.'] = 'Alabama St.'
    mapping['Bryant'] = 'Bryant'
    mapping['Clemson'] = 'Clemson'
    mapping['Georgia'] = 'Georgia'
    mapping['Kansas'] = 'Kansas'
    mapping['Missouri'] = 'Missouri'
    mapping['Montana'] = 'Montana'
    mapping['Oklahoma'] = 'Oklahoma'
    mapping['Troy'] = 'Troy'
    mapping['Utah St.'] = 'Utah St.'
    mapping['VCU'] = 'VCU'
    mapping['Vanderbilt'] = 'Vanderbilt'
    mapping['Wofford'] = 'Wofford'
    mapping['Xavier'] = 'Xavier'
    mapping['Yale'] = 'Yale'
    
    return mapping

class TournamentSimulator:
    def __init__(self, schedule_df, teams_df, team_mapping):
        self.schedule_df = schedule_df
        self.teams_df = teams_df
        self.team_mapping = team_mapping
        self.winners = {}  # {game_id: winning team}
        
        # Map round numbers to odds column names
        self.round_to_odds = {
            64: '64odds',
            32: '32odds',
            16: '16odds',
            8: '8odds',
            4: '4odds',
            2: '2odds'
        }
    
    def get_team_name(self, team):
        if team.startswith('Winner of:'):
            game_id = team.split(':')[1].strip()
            if game_id in self.winners:
                return self.winners[game_id]
            else:
                # If we haven't simulated this game yet, simulate it now
                return self.simulate_game(game_id)
        return self.team_mapping.get(team, team)
    
    def get_team_odds(self, team, round_num):
        team_row = self.teams_df[self.teams_df['teamName'] == team]
        if team_row.empty:
            return 0
        # Ensure round_num is an integer key if it's not already
        try:
            round_key = int(round_num) 
        except ValueError:
             round_key = 64 # Or handle error appropriately

        if round_key not in self.round_to_odds:
             odds_col = '64odds'
        else:
             odds_col = self.round_to_odds[round_key]

        odds_value = team_row[odds_col].iloc[0]
        return odds_value
    
    def simulate_game(self, game_id):
        game = self.schedule_df[self.schedule_df['gameID'] == game_id].iloc[0]
        round_num = game['round']
        team_a_raw = game['teamA']
        team_b_raw = game['teamB']
        
        # Resolve team names (this might recursively call simulate_game)
        team_a = self.get_team_name(team_a_raw)
        team_b = self.get_team_name(team_b_raw)
        
        # Debug Texas Tech game
        if game_id == 'R64_14' and DEBUG:
            print(f"DEBUG: Simulating Texas Tech game - Texas Tech vs UNC Wilmington")
            print(f"DEBUG: Team A: {team_a}, Team B: {team_b}")
        
        # Get odds for each team for the correct round
        odds_a = self.get_team_odds(team_a, round_num)
        odds_b = self.get_team_odds(team_b, round_num)
        
        # Debug Texas Tech game odds
        if game_id == 'R64_14' and DEBUG:
            print(f"DEBUG: Odds - {team_a}: {odds_a}, {team_b}: {odds_b}")
        
        # Determine winner based on odds (should be deterministic for 100/0)
        winner = None
        if odds_a == 0 and odds_b == 0:
            winner = random.choice([team_a, team_b])
        elif odds_a > 0 and odds_b == 0:
             winner = team_a
        elif odds_b > 0 and odds_a == 0:
             winner = team_b
        elif odds_a > 0 and odds_b > 0: # Both have > 0 odds? Should only be if both are 100?
            # Normalize odds and choose based on probability
            total_odds = odds_a + odds_b
            if total_odds == 0: # Should not happen if odds > 0
                winner = random.choice([team_a, team_b])
            else:
                prob_a = odds_a / total_odds
                rand_val = random.random() # Generate random number
                if rand_val < prob_a:
                    winner = team_a
                else:
                    winner = team_b
        else: # Should not happen if one is 100 and other is 0
            winner = random.choice([team_a, team_b])
        
        # Debug Texas Tech game winner
        if game_id == 'R64_14' and DEBUG:
            print(f"DEBUG: Winner of Texas Tech game: {winner}")
        
        # Record the winner
        self.winners[game_id] = winner
        return winner
    
    def get_games_for_date(self, date):
        return self.schedule_df[self.schedule_df['day'] == date]

class SurvivorPoolSimulator:
    def __init__(self, schedule_df, teams_df, picks_df, variance_factor, target_date):
        self.schedule_df = schedule_df
        self.teams_df = teams_df
        self.picks_df = picks_df
        self.variance_factor = variance_factor
        self.target_date = target_date
        
        # Create team mapping
        self.team_mapping = create_team_mapping(teams_df, picks_df, schedule_df)
        
        # Initialize tracking variables
        self.active_players = set(picks_df.index)
        self.eliminated_players = set()
        self.elimination_dates = {}  # {player_id: date eliminated}
        self.used_teams = defaultdict(set)  # {player_id: set of used teams}
        self.target_date_picks = {}  # {player_id: team picked on target date}
        self.target_date_triumphs = set()  # players who triumphed on target date
        self.winner = None
        self.winners = {}  # {game_id: winning team}
        
        # Process predetermined picks
        for player_id in self.active_players:
            self.used_teams[player_id] = set()
            for date in picks_df.columns:
                pick = picks_df.loc[player_id, date]
                if pd.notna(pick):
                    original_pick = pick
                    pick = self.team_mapping.get(pick, pick)
                    if pick != '':
                        self.used_teams[player_id].add(pick)
    
    def is_complete(self):
        return len(self.active_players - self.eliminated_players) <= 1 or self.winner is not None
    
    def get_player_pick(self, player_id, date, tournament_sim):
        if player_id in self.eliminated_players:
            # Debug for player 170
            if player_id == 170 and (date == '3/22/2025' or date == '3/23/2025') and DEBUG:
                print(f"DEBUG: Player 170 is already eliminated when checked on {date}")
            return None
            
        # Check if there's a predetermined pick
        pick_predetermined = None
        if date in self.picks_df.columns:
             pick_val = self.picks_df.get(date, pd.Series(index=self.picks_df.index)).loc[player_id]
             if pd.notna(pick_val):
                 original_pick = pick_val
                 mapped_pick = self.team_mapping.get(original_pick, original_pick)
                 if mapped_pick != '':
                     pick_predetermined = mapped_pick
                     # Debug for player 170
                     if player_id == 170 and (date == '3/22/2025' or date == '3/23/2025') and DEBUG:
                         print(f"DEBUG: Player 170 has predetermined pick {original_pick} mapped to {mapped_pick} on {date}")
                         print(f"DEBUG: Player 170 used teams before pick: {self.used_teams[player_id]}")
                     return pick_predetermined 
                 else:
                      pass # Predetermined pick maps to empty

        # Debug for player 170
        if player_id == 170 and (date == '3/22/2025' or date == '3/23/2025') and DEBUG:
            print(f"DEBUG: Player 170 has no predetermined pick on {date}, will simulate pick")
            print(f"DEBUG: Player 170 used teams before simulating: {self.used_teams[player_id]}")

        # --- Simulate Pick Logic --- 
        available_teams_all = self.get_available_teams(player_id)
        games = self.schedule_df[self.schedule_df['day'] == date]
        playing_teams = set()
        current_round = None
        for _, game in games.iterrows():
            team_a = self.team_mapping.get(game['teamA'], game['teamA'])
            team_b = self.team_mapping.get(game['teamB'], game['teamB'])
            if team_a.startswith('Winner of:'):
                game_id = team_a.split(':')[1].strip()
                if game_id in tournament_sim.winners:
                    team_a = tournament_sim.winners[game_id]
            if team_b.startswith('Winner of:'):
                game_id = team_b.split(':')[1].strip()
                if game_id in tournament_sim.winners:
                    team_b = tournament_sim.winners[game_id]
            if isinstance(team_a, str) and team_a: playing_teams.add(team_a)
            if isinstance(team_b, str) and team_b: playing_teams.add(team_b)
            current_round = game['round']
        
        available_teams_playing = [team for team in available_teams_all if team in playing_teams]

        # Debug for player 170
        if player_id == 170 and (date == '3/22/2025' or date == '3/23/2025') and DEBUG:
            print(f"DEBUG: Player 170 available teams playing on {date}: {available_teams_playing}")
            print(f"DEBUG: All teams playing on {date}: {playing_teams}")

        if not available_teams_playing:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            if player_id == 170 and (date == '3/22/2025' or date == '3/23/2025') and DEBUG:
                print(f"DEBUG: Player 170 eliminated on {date} - no available teams to pick from")
            return None
            
        # Get odds for available teams
        team_odds = {}
        for team in available_teams_playing:
            team_row = self.teams_df[self.teams_df['teamName'] == team]
            if not team_row.empty:
                seed = team_row['seed'].iloc[0]
                odds = tournament_sim.get_team_odds(team, current_round)
                boost_multiplier = SEED_BOOSTS.get(str(current_round), {}).get(seed, 1)
                pick_weight = odds * boost_multiplier
                team_odds[team] = pick_weight
        
        if not team_odds:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            return None
            
        # Always use probabilistic selection based on the weights
        teams = list(team_odds.keys())
        weights = list(team_odds.values())
        
        if self.variance_factor > 0:
            # Apply variance factor adjustment if variance_factor > 0
            # Normalize to get base probabilities
            total_weight = sum(weights)
            
            # If total weight is zero, eliminate player
            if total_weight <= 0:
                if DEBUG:
                    print(f"DEBUG: Player {player_id} eliminated on {date} - all weights zero")
                self.eliminated_players.add(player_id)
                self.elimination_dates[player_id] = date
                return None
                
            base_probs = [w / total_weight for w in weights]
            
            # Apply variance transformation to flatten probability distribution
            adjusted_probs = []
            for prob in base_probs:
                # Higher variance_factor makes distribution more uniform
                adjusted_prob = prob ** (1 / (1 + self.variance_factor))
                adjusted_probs.append(adjusted_prob)
                
            # Re-normalize the adjusted probabilities
            total_adjusted = sum(adjusted_probs)
            weights = [p / total_adjusted for p in adjusted_probs]
        else:
            # Check if all weights are zero even without variance adjustment
            if sum(weights) <= 0:
                if DEBUG:
                    print(f"DEBUG: Player {player_id} eliminated on {date} - all weights zero")
                self.eliminated_players.add(player_id)
                self.elimination_dates[player_id] = date
                return None
            
        # Make the random selection based on weights (original or adjusted)
        pick_simulated = random.choices(teams, weights=weights, k=1)[0]
        
        self.used_teams[player_id].add(pick_simulated)
        return pick_simulated
    
    def get_available_teams(self, player_id):
        used = self.used_teams[player_id]
        return [team for team in self.teams_df['teamName'] if team not in used]
        
    def get_seed_sum(self, player_id):
        """Calculate sum of seeds for all teams picked by a player"""
        total = 0
        for team in self.used_teams[player_id]:
            team_row = self.teams_df[self.teams_df['teamName'] == team]
            if not team_row.empty:
                total += team_row['seed'].iloc[0]
        return total
        
    def determine_winner_from_candidates(self, candidate_players):
        """Determines the winner from a list of candidates using tiebreakers."""
        if not candidate_players:
            return None
        if len(candidate_players) == 1:
            return candidate_players[0]
            
        # First tiebreaker: sum of seeds
        seed_sums = {player: self.get_seed_sum(player) for player in candidate_players}
        max_sum = max(seed_sums.values())
        players_with_max_sum = [p for p in candidate_players if seed_sums[p] == max_sum]
        
        if len(players_with_max_sum) == 1:
            return players_with_max_sum[0]
            
        # Second tiebreaker: random selection
        # return sorted(players_with_max_sum)[0] # Use lowest ID for deterministic tiebreak for now
        return random.choice(players_with_max_sum) # Use random selection as specified in the rules

    def run_simulation(self, tournament_sim):
        # Simulate until pool is complete
        current_date = min(self.schedule_df['day'])
        while not self.is_complete():
            # Add debug to see which date we're processing
            if DEBUG:
                print(f"DEBUG: Processing date {current_date}")
            
            # Check if any player has a predetermined pick for this date
            # If at least one player has a pick, those without picks should be eliminated
            if current_date in self.picks_df.columns:
                players_with_picks = set(self.picks_df[pd.notna(self.picks_df[current_date])].index)
                if players_with_picks and DEBUG:  
                    print(f"DEBUG: {len(players_with_picks)} players have predetermined picks on {current_date}")
                    # Debug specific players of interest
                    for player_id in [1548, 1881]:
                        if player_id in players_with_picks:
                            print(f"DEBUG: Player {player_id} has a predetermined pick on {current_date}")
                        elif player_id in self.active_players and player_id not in self.eliminated_players:
                            print(f"DEBUG: Player {player_id} is active but has no predetermined pick on {current_date}")
                        elif player_id in self.eliminated_players:
                            print(f"DEBUG: Player {player_id} is already eliminated prior to {current_date}")
                
                if players_with_picks:  # Only check if at least one player has a pick
                    players_without_picks = self.active_players - self.eliminated_players - players_with_picks
                    if DEBUG and players_without_picks:
                        print(f"DEBUG: {len(players_without_picks)} players will be eliminated for not having picks on {current_date}")
                        if 1548 in players_without_picks:
                            print(f"DEBUG: Player 1548 will be eliminated for not having a pick on {current_date}")
                        if 1881 in players_without_picks:
                            print(f"DEBUG: Player 1881 will be eliminated for not having a pick on {current_date}")
                    
                    for player_id in players_without_picks:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        if DEBUG and (player_id == 1548 or player_id == 1881 or len(players_without_picks) < 10):
                            print(f"DEBUG: Player {player_id} eliminated for not having a predetermined pick on {current_date} when others did")
            
            # Special handling for combined days 7 and 8
            if current_date in ['3/29/2025', '3/30/2025']:
                # Get games for both days
                games_day7 = tournament_sim.get_games_for_date('3/29/2025')
                games_day8 = tournament_sim.get_games_for_date('3/30/2025')
                
                # Combine games from both days
                games = pd.concat([games_day7, games_day8])
                games = games.sort_values('round')
                
                # First check if any players have predetermined picks for either day
                day1_has_picks = False
                day2_has_picks = False
                for player_id in self.active_players:
                    if player_id not in self.eliminated_players:
                        # Check for predetermined picks
                        if '3/29/2025' in self.picks_df.columns:
                            pick_val = self.picks_df.get('3/29/2025', pd.Series(index=self.picks_df.index)).loc[player_id]
                            if pd.notna(pick_val) and self.team_mapping.get(pick_val, pick_val) != '':
                                day1_has_picks = True
                        if '3/30/2025' in self.picks_df.columns:
                            pick_val = self.picks_df.get('3/30/2025', pd.Series(index=self.picks_df.index)).loc[player_id]
                            if pd.notna(pick_val) and self.team_mapping.get(pick_val, pick_val) != '':
                                day2_has_picks = True
                        if day1_has_picks and day2_has_picks:
                            break
                
                if DEBUG:
                    print(f"DEBUG: Combined days check - Day 1 has predetermined picks: {day1_has_picks}, Day 2 has predetermined picks: {day2_has_picks}")
                
                # Track which players are active before any games are played
                active_players_on_date = set()
                player_picks = {}  # {player_id: [pick1, pick2]}
                
                for player_id in self.active_players:
                    if player_id not in self.eliminated_players:
                        # Get picks for both days
                        pick1 = self.get_player_pick(player_id, '3/29/2025', tournament_sim)
                        pick2 = self.get_player_pick(player_id, '3/30/2025', tournament_sim)
                        
                        # Check if player has predetermined picks for both days when others do
                        pick1_predetermined = False
                        pick2_predetermined = False
                        if '3/29/2025' in self.picks_df.columns:
                            pick_val = self.picks_df.get('3/29/2025', pd.Series(index=self.picks_df.index)).loc[player_id]
                            if pd.notna(pick_val) and self.team_mapping.get(pick_val, pick_val) != '':
                                pick1_predetermined = True
                        if '3/30/2025' in self.picks_df.columns:
                            pick_val = self.picks_df.get('3/30/2025', pd.Series(index=self.picks_df.index)).loc[player_id]
                            if pd.notna(pick_val) and self.team_mapping.get(pick_val, pick_val) != '':
                                pick2_predetermined = True
                        
                        # Eliminate player if they don't have predetermined picks for both days when others do
                        if (day1_has_picks and not pick1_predetermined) or (day2_has_picks and not pick2_predetermined):
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            if DEBUG:
                                print(f"DEBUG: Player {player_id} eliminated for not having predetermined picks for both days (has day1: {pick1_predetermined}, has day2: {pick2_predetermined})")
                            continue
                        
                        if pick1 is not None and pick2 is not None:
                            active_players_on_date.add(player_id)
                            player_picks[player_id] = [pick1, pick2]
                            # Record picks if this is the target date
                            if '3/29/2025' == self.target_date or '3/30/2025' == self.target_date:
                                self.target_date_picks[player_id] = f"{pick1}, {pick2}"
                        else:
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            if DEBUG:
                                print(f"DEBUG: Player {player_id} eliminated in combined day - pick1: {pick1}, pick2: {pick2}")
                
                # Simulate each game
                for _, game in games.iterrows():
                    # Simulate game
                    winner = tournament_sim.simulate_game(game['gameID'])
                    self.winners[game['gameID']] = winner
                    
                    # Process each active player's picks
                    for player_id in list(active_players_on_date):
                        if player_id in self.eliminated_players:
                            # Debug for player 170
                            if player_id == 170 and (current_date == '3/22/2025' or current_date == '3/23/2025') and DEBUG:
                                print(f"DEBUG: Player 170 already eliminated during game processing on {current_date}")
                            continue
                            
                        picks = player_picks.get(player_id)
                        if picks is None:
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            # Debug for player 170
                            if player_id == 170 and (current_date == '3/22/2025' or current_date == '3/23/2025') and DEBUG:
                                print(f"DEBUG: Player 170 eliminated during game processing - no pick on {current_date}")
                            continue
                            
                        # Get the teams playing in this game
                        team_a = tournament_sim.get_team_name(game['teamA'])
                        team_b = tournament_sim.get_team_name(game['teamB'])
                        
                        # Check if either of the player's picks lost in this game
                        if (picks[0] in [team_a, team_b] and picks[0] != winner) or \
                           (picks[1] in [team_a, team_b] and picks[1] != winner):
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            # Debug for player 1881 or player 170
                            if (player_id == 1881 or player_id == 170) and DEBUG:
                                print(f"DEBUG: Player {player_id} eliminated during combined day processing - picked {picks[0]} and {picks[1]}, but team {picks[0] if picks[0] in [team_a, team_b] and picks[0] != winner else picks[1]} lost to {winner}")
                                print(f"DEBUG: Game was between {team_a} and {team_b}")
                            # Debug if this is the Texas Tech game
                            if game['gameID'] == 'R64_14' and DEBUG:
                                print(f"DEBUG: Player {player_id} picked {picks} and was eliminated (winner was {winner})")
                
                # Move to next date after the combined days
                current_date = get_next_date('3/30/2025', self.schedule_df)
                continue
            
            # Normal processing for other days
            games = tournament_sim.get_games_for_date(current_date)
            games = games.sort_values('round')
            
            # Extra debug for 3/27
            if current_date == '3/27/2025' and DEBUG:
                print(f"DEBUG: Finding games for 3/27/2025, found {len(games)} games")
                print(f"DEBUG: Checking available teams and eliminations for 3/27/2025")
            
            # Track which players are active before any games are played
            active_players_on_date = set()
            player_picks = {}  # {player_id: pick}
            
            for player_id in self.active_players:
                if player_id not in self.eliminated_players:
                    # Pass tournament_sim to get_player_pick
                    pick = self.get_player_pick(player_id, current_date, tournament_sim)
                    # Extra debug for 3/27
                    if current_date == '3/27/2025' and player_id < 10 and DEBUG:
                        available_teams = self.get_available_teams(player_id)
                        playing_teams = set()
                        for _, game in games.iterrows():
                            team_a = self.team_mapping.get(game['teamA'], game['teamA'])
                            if team_a.startswith('Winner of:'):
                                game_id = team_a.split(':')[1].strip()
                                if game_id in tournament_sim.winners:
                                    team_a = tournament_sim.winners[game_id]
                            team_b = self.team_mapping.get(game['teamB'], game['teamB'])
                            if team_b.startswith('Winner of:'):
                                game_id = team_b.split(':')[1].strip()
                                if game_id in tournament_sim.winners:
                                    team_b = tournament_sim.winners[game_id]
                            if isinstance(team_a, str) and team_a: playing_teams.add(team_a)
                            if isinstance(team_b, str) and team_b: playing_teams.add(team_b)
                        available_playing = [t for t in available_teams if t in playing_teams]
                        print(f"DEBUG: Player {player_id} - Available teams: {len(available_teams)}, Playing today: {len(playing_teams)}, Available+Playing: {len(available_playing)}")
                        if not available_playing:
                            print(f"DEBUG: Player {player_id} has no available teams playing on 3/27/2025 and will be eliminated")
                        print(f"DEBUG: Used teams for player {player_id}: {self.used_teams[player_id]}")
                        if playing_teams:
                            print(f"DEBUG: Teams playing on 3/27/2025 include: {list(playing_teams)[:5]}...")
                        
                    if pick is not None:
                        active_players_on_date.add(player_id)
                        player_picks[player_id] = pick
                        if current_date == self.target_date:
                            self.target_date_picks[player_id] = pick
                    # If pick is None, player is eliminated (handled inside get_player_pick)
            
            # Debug players and their picks
            if (current_date == '3/20/2025' or current_date == '3/27/2025') and DEBUG:
                print(f"DEBUG: Players and picks on {current_date}:")
                for player_id, pick in player_picks.items():
                    if player_id < 10:  # Only show first 10 players to keep output manageable
                        print(f"DEBUG: Player {player_id} picked {pick}")
                print(f"DEBUG: Active players count before games: {len(active_players_on_date)}")
            
            # Simulate each game
            for _, game in games.iterrows():
                # Simulate game
                winner = tournament_sim.simulate_game(game['gameID'])
                self.winners[game['gameID']] = winner
                
                # Debug for game R64_14 (Texas Tech game)
                if game['gameID'] == 'R64_14' and DEBUG:
                    print(f"DEBUG: After simulating game R64_14:")
                    eliminated_count_before = len(self.eliminated_players)
                
                # Process each active player's pick
                for player_id in list(active_players_on_date):
                    if player_id in self.eliminated_players:
                        # Debug for player 170
                        if player_id == 170 and (current_date == '3/22/2025' or current_date == '3/23/2025') and DEBUG:
                            print(f"DEBUG: Player 170 already eliminated during game processing on {current_date}")
                        continue
                        
                    pick = player_picks.get(player_id)
                    if pick is None:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        # Debug for player 170
                        if player_id == 170 and (current_date == '3/22/2025' or current_date == '3/23/2025') and DEBUG:
                            print(f"DEBUG: Player 170 eliminated during game processing - no pick on {current_date}")
                        continue
                        
                    # Get the teams playing in this game
                    team_a = tournament_sim.get_team_name(game['teamA'])
                    team_b = tournament_sim.get_team_name(game['teamB'])
                    
                    # Only eliminate player if their picked team lost in this game
                    if pick in [team_a, team_b] and pick != winner:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        # Debug for player 170
                        if player_id == 170 and (current_date == '3/22/2025' or current_date == '3/23/2025') and DEBUG:
                            print(f"DEBUG: Player 170 eliminated during game processing - picked {pick} but {winner} won on {current_date}")
                            print(f"DEBUG: Game was between {team_a} and {team_b}")
                        # Debug if this is the Texas Tech game
                        if game['gameID'] == 'R64_14' and DEBUG:
                            print(f"DEBUG: Player {player_id} picked {pick} and was eliminated (winner was {winner})")
                
                # Debug after processing Texas Tech game
                if game['gameID'] == 'R64_14' and DEBUG:
                    eliminated_count_after = len(self.eliminated_players)
                    print(f"DEBUG: Eliminated {eliminated_count_after - eliminated_count_before} players after game R64_14")
                    print(f"DEBUG: Total eliminated players now: {eliminated_count_after}")
                    print(f"DEBUG: Active players left: {len(self.active_players) - eliminated_count_after}")
            
            # Check if we have a winner after all games for the day
            active_count = len(self.active_players - self.eliminated_players)
            if DEBUG:
                print(f"DEBUG: After all games on {current_date}, active players count: {active_count}")
            
            if active_count == 0:
                # All players have been eliminated, determine winner based on who lasted longest
                self.winner = self.determine_winner_from_candidates(list(self.active_players - self.eliminated_players))
                if DEBUG:
                    print(f"DEBUG: All players eliminated, winner based on tiebreaker: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            elif active_count == 1:
                # One player still active, they win
                self.winner = list(self.active_players - self.eliminated_players)[0]
                if DEBUG:
                    print(f"DEBUG: One player remaining, winner: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            
            current_date = get_next_date(current_date, self.schedule_df)
            if current_date is None:
                # Reached end of schedule, determine winner based on who lasted longest
                self.winner = self.determine_winner_from_candidates(list(self.active_players - self.eliminated_players))
                if DEBUG:
                    print(f"DEBUG: End of schedule reached, winner: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break

        # --- Determine Winner AFTER loop finishes --- 
        final_active_players = list(self.active_players - self.eliminated_players)
        if DEBUG:
            print(f"DEBUG: Final active players count: {len(final_active_players)}")
        
        if len(final_active_players) >= 1:
            # If one or more players survived the whole tournament
            self.winner = self.determine_winner_from_candidates(final_active_players)
            if DEBUG:
                print(f"DEBUG: Final winner (players survived): {self.winner}")
        elif not self.elimination_dates:
            # No one eliminated, no one survived? Edge case - should not happen
            self.winner = None 
            if DEBUG:
                print(f"DEBUG: No winner determined (no elimination dates)")
        else:
            # All players were eliminated at some point, find the last ones standing
            # Need to group by date, not by player ID
            players_by_date = defaultdict(list)
            for player_id, date in self.elimination_dates.items():
                players_by_date[date].append(player_id)
            
            # Debug elimination dates
            if DEBUG:
                print(f"DEBUG: Elimination dates: {set(players_by_date.keys())}")
            
            if players_by_date:
                latest_date = max(players_by_date.keys())
                last_players_eliminated = players_by_date[latest_date]
                if DEBUG:
                    print(f"DEBUG: Latest date: {latest_date}, Players: {len(last_players_eliminated)}")
                self.winner = self.determine_winner_from_candidates(last_players_eliminated)
                if DEBUG:
                    print(f"DEBUG: Final winner (all eliminated on {latest_date}): {self.winner}")
                    print(f"DEBUG: Player count eliminated on last day: {len(last_players_eliminated)}")
            else:
                if DEBUG:
                    print(f"DEBUG: No players_by_date found!")
                self.winner = None
            
        # Final debug of winner
        if DEBUG:
            print(f"DEBUG: FINAL WINNER: {self.winner}")
            # Debug the triumphs
            print(f"DEBUG: Target date triumphs: {self.target_date_triumphs}")
            print(f"DEBUG: Target date picks count: {len(self.target_date_picks)}")

def get_next_date(current_date, schedule_df):
    """Get the next date in the schedule after current_date."""
    dates = sorted(schedule_df['day'].unique())
    current_idx = dates.index(current_date)
    if current_idx + 1 < len(dates):
        return dates[current_idx + 1]
    return None

def run_single_simulation(sim_number, schedule_df, teams_df, picks_df, team_mapping, target_date, target_player, variance_factor):
    """Run a single simulation and return the results"""
    # We don't print the simulation number here anymore if using multiprocessing
    # print(f"Running simulation {sim_number+1}") 
    
    # Create new simulators for this simulation
    tournament_sim = TournamentSimulator(schedule_df.copy(), teams_df.copy(), team_mapping.copy()) # Use copies to ensure isolation
    survivor_sim = SurvivorPoolSimulator(schedule_df.copy(), teams_df.copy(), picks_df.copy(), variance_factor, target_date) # Use copies
    
    # Run the simulation
    survivor_sim.run_simulation(tournament_sim)
    
    # Winner is now correctly determined inside run_simulation
    final_winner_id = survivor_sim.winner
    if DEBUG:
        print(f"DEBUG RUN_SINGLE_SIMULATION: Final pool winner ID: {final_winner_id}")

    # Make sure we simulate the final game if it hasn't been simulated yet
    # It might be necessary to simulate *all* games to get the full bracket outcome
    all_game_ids = schedule_df['gameID'].unique()
    for game_id in all_game_ids:
        if game_id not in tournament_sim.winners:
            try:
                # Attempt to simulate any remaining games
                tournament_sim.simulate_game(game_id)
            except Exception:
                # Handle cases where games can't be simulated yet (e.g., waiting for prior rounds)
                # This loop ensures we try to complete the bracket as much as possible
                pass 

    # Prepare results to return
    results = {
        'tournament_winner': tournament_sim.winners.get('FINAL', None), # Get final winner specifically
        'full_tournament_outcome': tournament_sim.winners.copy(), # Capture all game winners
        'player_picks': survivor_sim.target_date_picks.copy(),
        'final_pool_winner': final_winner_id, # Store the actual pool winner
        'target_player_pick': survivor_sim.target_date_picks.get(target_player),
    }
        
    return results

def run_simulations(num_simulations, target_date, target_player, variance_factor, use_multiprocessing=True, num_processes=None):
    # Load data once
    schedule_df, teams_df, picks_df, team_mapping = load_data()
    
    # Initialize stats (ensure triumph is 0)
    player_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0})
    team_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0}) # Add triumphs back
    target_player_team_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0}) # Add triumphs back
    tournament_winners = defaultdict(int)
    pool_winners_count = defaultdict(int)
    all_tournament_outcomes = [] 

    if use_multiprocessing and num_simulations > 1:
        # Determine number of processes to use
        if num_processes is None:
            num_processes = min(multiprocessing.cpu_count(), num_simulations)
        
        print(f"Running {num_simulations} simulations using {num_processes} processes...")
        
        # Create a partial function with fixed arguments
        # Pass copies of dataframes to ensure process isolation
        worker_func = partial(
            run_single_simulation,
            schedule_df=schedule_df.copy(),
            teams_df=teams_df.copy(),
            picks_df=picks_df.copy(),
            team_mapping=team_mapping.copy(),
            target_date=target_date,
            target_player=target_player,
            variance_factor=variance_factor
        )
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Map the worker function to the range of simulation indices
            # Using imap_unordered for potentially better memory usage with many sims
            results_iterator = pool.imap_unordered(worker_func, range(num_simulations))
            
            # Process results as they complete
            processed_count = 0
            for sim_result in results_iterator:
                processed_count += 1
                print(f"Processing result {processed_count}/{num_simulations}", end='\r') # Progress indicator
                
                # Aggregate tournament winner
                if sim_result['tournament_winner']: tournament_winners[sim_result['tournament_winner']] += 1
                all_tournament_outcomes.append(sim_result['full_tournament_outcome'])
                
                # Aggregate pool winner
                actual_winner_id = sim_result['final_pool_winner']
                if actual_winner_id is not None:
                    pool_winners_count[actual_winner_id] += 1

                # Get the winner's pick on the target date, if any
                winner_pick_on_target = sim_result['player_picks'].get(actual_winner_id)

                # Process player picks made on the target_date
                for player_id, pick in sim_result['player_picks'].items():
                    player_stats[player_id]['picks'] += 1
                    # Aggregate team picks (all players)
                    if isinstance(pick, str) and ',' in pick:
                        picks_list = [p.strip() for p in pick.split(',')]
                        for team in picks_list:
                            # Ensure team name is valid
                            if team:
                                team_stats[team]['picks'] += 1
                                if DEBUG:
                                    print(f"DEBUG: Added pick for team '{team}' (from '{pick}')")
                            else:
                                if DEBUG:
                                    print(f"DEBUG: Player {player_id} has empty/invalid pick: '{pick}'")
                    elif pick:
                        # Add direct pick handling for non-comma case
                        team_stats[pick]['picks'] += 1
                        if DEBUG:
                            print(f"DEBUG: Added pick for team '{pick}' (direct)")
                    else:
                        if DEBUG:
                            print(f"DEBUG: Player {player_id} has empty/invalid pick: '{pick}'")
                    
                    # Add player triumph count ONLY to the actual winner
                    if player_id == actual_winner_id:
                        player_stats[player_id]['triumphs'] += 1
                
                # Process target player's pick - add picks and check if team won
                if sim_result['target_player_pick']:
                    pick = sim_result['target_player_pick']
                    # Check if pick is a string before splitting
                    if isinstance(pick, str) and ',' in pick:
                        picks_list = [p.strip() for p in pick.split(',')]
                        for team in picks_list:
                            # Ensure team name is valid
                            if team:
                                target_player_team_stats[team]['picks'] += 1
                                # Only count as triumph if target player won the entire pool
                                if target_player == sim_result['final_pool_winner']:
                                    target_player_team_stats[team]['triumphs'] += 1
                    elif pick:
                        target_player_team_stats[pick]['picks'] += 1
                        # Only count as triumph if target player won the entire pool
                        if target_player == sim_result['final_pool_winner']:
                            target_player_team_stats[pick]['triumphs'] += 1

                # Add team triumph counts if the winner picked on the target date
                if winner_pick_on_target:
                    # Team Stats Triumph
                    if isinstance(winner_pick_on_target, str) and ',' in winner_pick_on_target:
                        picks_list = [p.strip() for p in winner_pick_on_target.split(',')]
                        for team in picks_list: team_stats[team]['triumphs'] += 1
                    elif winner_pick_on_target: 
                        team_stats[winner_pick_on_target]['triumphs'] += 1

            print("\nResults processing complete.") # Newline after progress indicator
    else:
        print(f"Running {num_simulations} simulations sequentially...")
        # Original sequential implementation
        for sim in range(num_simulations):
            print(f"Running simulation {sim + 1}/{num_simulations}")
            
            # Run single simulation logic directly
            sim_result = run_single_simulation(
                sim, schedule_df, teams_df, picks_df, team_mapping, 
                target_date, target_player, variance_factor
            )

            # Aggregate tournament winner
            if sim_result['tournament_winner']: tournament_winners[sim_result['tournament_winner']] += 1
            all_tournament_outcomes.append(sim_result['full_tournament_outcome'])

            # Aggregate pool winner
            actual_winner_id = sim_result['final_pool_winner']
            if actual_winner_id is not None:
                pool_winners_count[actual_winner_id] += 1

            # Get the winner's pick on the target date, if any
            winner_pick_on_target = sim_result['player_picks'].get(actual_winner_id)
            
            # Process player picks made on the target_date
            for player_id, pick in sim_result['player_picks'].items():
                 player_stats[player_id]['picks'] += 1
                 # Aggregate team picks (all players)
                 if isinstance(pick, str) and ',' in pick:
                     picks_list = [p.strip() for p in pick.split(',')]
                     for team in picks_list:
                         # Ensure team name is valid
                         if team:
                             team_stats[team]['picks'] += 1
                             if DEBUG:
                                 print(f"DEBUG: Added pick for team '{team}' (from '{pick}')")
                 elif pick:
                     # Add direct pick handling for non-comma case
                     team_stats[pick]['picks'] += 1
                     if DEBUG:
                         print(f"DEBUG: Added pick for team '{pick}' (direct)")
                 else:
                     if DEBUG:
                         print(f"DEBUG: Player {player_id} has empty/invalid pick: '{pick}'")
                 
                 # Add player triumph count ONLY to the actual winner
                 if player_id == actual_winner_id:
                    player_stats[player_id]['triumphs'] += 1

            # Debug team stats collection
            if DEBUG:
                pick_count = sum(team_stats[team]['picks'] for team in team_stats)
                team_count = sum(1 for team in team_stats if team_stats[team]['picks'] > 0)
                print(f"DEBUG: Collected {pick_count} total team picks across {team_count} teams")

            # Process target player's pick - add picks and check if team won
            if sim_result['target_player_pick']:
                 pick = sim_result['target_player_pick']
                 # Check if pick is a string before splitting
                 if isinstance(pick, str) and ',' in pick:
                     picks_list = [p.strip() for p in pick.split(',')]
                     for team in picks_list:
                          # Ensure team name is valid
                          if team:
                              target_player_team_stats[team]['picks'] += 1
                              # Only count as triumph if target player won the entire pool
                              if target_player == sim_result['final_pool_winner']:
                                  target_player_team_stats[team]['triumphs'] += 1
                 elif pick:
                      target_player_team_stats[pick]['picks'] += 1
                      # Only count as triumph if target player won the entire pool
                      if target_player == sim_result['final_pool_winner']:
                          target_player_team_stats[pick]['triumphs'] += 1

            # Add team triumph counts if the winner picked on the target date
            if winner_pick_on_target:
                # Team Stats Triumph
                if isinstance(winner_pick_on_target, str) and ',' in winner_pick_on_target:
                    picks_list = [p.strip() for p in winner_pick_on_target.split(',')]
                    for team in picks_list: team_stats[team]['triumphs'] += 1
                elif winner_pick_on_target: 
                    team_stats[winner_pick_on_target]['triumphs'] += 1

    # --- Determinism Check ---
    if DEBUG:
        print("\n--- Tournament Determinism Check ---")
        if num_simulations > 1 and all_tournament_outcomes:
            first_outcome = all_tournament_outcomes[0]
            # Convert dictionaries to sorted list of items for comparison
            first_outcome_sorted = sorted(first_outcome.items())
            are_identical = True
            diff_found_at = -1
            for i in range(1, len(all_tournament_outcomes)):
                current_outcome_sorted = sorted(all_tournament_outcomes[i].items())
                if current_outcome_sorted != first_outcome_sorted:
                    are_identical = False
                    diff_found_at = i
                    break # Stop at first difference
                
            if are_identical:
                print("Result: All simulated tournament outcomes were IDENTICAL.")
            else:
                print(f"Result: DIFFERENCES FOUND between tournament outcomes.")
                print(f"Discrepancy first detected between simulation 1 and simulation {diff_found_at + 1}.")
                # Optional: Print the differing dictionaries for debugging
                # print("\\nOutcome 1:")
                # print(dict(first_outcome_sorted))
                # print(f"\\nOutcome {diff_found_at + 1}:")
                # print(dict(sorted(all_tournament_outcomes[diff_found_at].items())))
        elif num_simulations <= 1:
            print("Result: Only one simulation run, cannot check for determinism.")
        else:
             print("Result: No tournament outcomes collected to check.")
        print("------------------------------------")

    # --- Missing Player Check ---
    if DEBUG:
        print("\n--- Missing Player Check ---")
        all_player_ids = set(picks_df.index) # Get all players from the input picks file
        players_who_picked_on_target = set(player_stats.keys()) # Players who appeared in stats for the target date
        
        # Find players in the input file who didn't end up in the target date stats
        missing_players = all_player_ids - players_who_picked_on_target
        
        if not missing_players:
            print(f"Result: All {len(all_player_ids)} players from sample_picks.csv made a pick on {target_date}.")
        else:
            print(f"Result: {len(missing_players)} players from sample_picks.csv did NOT make a recorded pick on {target_date}.")
            # Filter missing players to only those who actually had a non-null pick specified for the target date in the input CSV
            players_with_valid_input_pick = set(picks_df[pd.notna(picks_df[target_date])].index)
            truly_missing_with_input = sorted(list(missing_players.intersection(players_with_valid_input_pick)))
            
            if truly_missing_with_input:
                 print("Players expected to pick based on input CSV but missing from stats:")
                 print(truly_missing_with_input)
            else:
                print("All players missing from stats also had no valid pick specified in sample_picks.csv for this date.")
                
            # Optionally, list players missing for other reasons (e.g., eliminated before target date)
            # missing_for_other_reasons = sorted(list(missing_players - players_with_valid_input_pick))
            # if missing_for_other_reasons:
            #    print("\nPlayers missing from stats (likely eliminated before target date or had null input pick):")
            #    print(missing_for_other_reasons)

        print("---------------------------")

    # Modify print_statistics to show the overall pool winner distribution
    # (Instead of flawed target_date_triumphs)
    print("\n--- Overall Survivor Pool Winner Counts ---")
    if pool_winners_count:
        for player_id, wins in sorted(pool_winners_count.items()):
            print(f"  Player {player_id}: {wins} wins ({wins/num_simulations*100:.1f}%)")
    else:
        print("  No pool winner recorded across simulations.")
    print("-----------------------------------------")

    # Pass pool_winners_count to print_statistics if needed for player table?
    # Or remove the flawed triumph column from print_statistics
    print_statistics(num_simulations, player_stats, team_stats, target_player_team_stats, tournament_winners, target_player, teams_df)

def print_statistics(num_simulations, player_stats, team_stats, target_player_team_stats, tournament_winners, target_player, teams_df):
    """Print all statistics tables with formatting and team seeds."""
    # Team stats summary - removed debug flag to always show this info
    print("\nTeam Statistics Summary:")
    print(f"Total teams with picks: {len(team_stats)}")
    print(f"Total picks recorded: {sum(team_stats[team]['picks'] for team in team_stats)}")
    if len(team_stats) == 0:
        print("WARNING: No team statistics data available!")
    
    # Player Statistics (Restore Triumph Probability)
    print("\nPlayer Statistics (Based on Target Date Picks & Overall Pool Wins):")
    print("Player ID | Picks on Target Date | Total Triumphs | Triumph Probability")
    print("-" * 75)
    total_target_date_picks = 0
    total_pool_wins = 0
    for player_id in sorted(player_stats.keys()):
        stats = player_stats[player_id]
        if stats['picks'] > 0: # Only show players who picked on target date
            picks_on_target = stats['picks']
            wins = stats.get('triumphs', 0) 
            win_prob = (wins / num_simulations * 100) 
            print(f"{player_id:9d} | {picks_on_target:20d} | {wins:15d} | {win_prob:10.2f}%")
            total_target_date_picks += picks_on_target
            total_pool_wins += wins 
    print("-" * 75)
    avg_win_prob = (total_pool_wins / total_target_date_picks * 100) if total_target_date_picks > 0 else 0
    print(f"TOTAL     | {total_target_date_picks:20d} | {total_pool_wins:15d} | {avg_win_prob:10.2f}%")
    
    # Team Statistics (All Players) - Restore Triumph columns
    print("\nTeam Statistics (All Players, Based on Target Date Picks):")
    team_width = 25
    picks_width = 20
    pct_width = 12
    triumphs_width = 10 # Add back
    prob_width = 18 # Adjust width for header
    headers = (f"{'Team':^{team_width}} | "
               f"{'Picks on Target Date':^{picks_width}} | "
               f"{'% of Picks':^{pct_width}} | "
               f"{'Triumphs':^{triumphs_width}} | " # Add back
               f"{'Triumph Probability':^{prob_width}}") # Add back
    separator = "-" * len(headers)
    print(headers)
    print(separator)
    total_picks = sum(team_stats[team]['picks'] for team in team_stats)
    total_triumphs = sum(team_stats[team]['triumphs'] for team in team_stats) # Recalculate based on new aggregation
    # Overall triumph probability for the table (triumphs / total picks on this day)
    total_prob = (total_triumphs / total_picks * 100) if total_picks > 0 else 0
    
    # Show all teams with at least one pick, regardless of triumphs
    for team in sorted(team_stats.keys()):
        picks = team_stats[team]['picks']
        # Only show teams that were actually picked
        if picks > 0:
            triumphs = team_stats[team].get('triumphs', 0) # Use .get() for safety
            pick_pct = (picks / total_picks * 100) if total_picks > 0 else 0
            prob = (triumphs / picks * 100) if picks > 0 else 0 # Per-team triumph probability
            seed = teams_df[teams_df['teamName'] == team]['seed'].iloc[0] if not teams_df[teams_df['teamName'] == team].empty else None
            team_display = team if seed is None else f"{team} ({seed})"
            print(f"{team_display:<{team_width}} | "
                  f"{picks:>{picks_width}} | "
                  f"{pick_pct:>{pct_width-1}.2f}% | "
                  f"{triumphs:>{triumphs_width}} | " # Add back
                  f"{prob:>{prob_width-1}.2f}%" ) # Add back
    print(separator)
    print(f"{'TOTAL':<{team_width}} | "
          f"{total_picks:>{picks_width}} | "
          f"{100:>{pct_width-1}.2f}% | "
          f"{total_triumphs:>{triumphs_width}} | " # Add back
          f"{total_prob:>{prob_width-1}.2f}%" ) # Add back
    
    # Team Statistics (Target Player)
    print(f"\nTeam Statistics (Player {target_player}, Based on Target Date Picks):")
    # Print debug info about target player picks
    print(f"Target player has {sum(target_player_team_stats[team]['picks'] for team in target_player_team_stats)} total picks across {len(target_player_team_stats)} teams")
    
    # Reuse headers and separator defined above
    print(headers) 
    print(separator)
    total_picks = sum(target_player_team_stats[team]['picks'] for team in target_player_team_stats)
    total_triumphs = sum(target_player_team_stats[team]['triumphs'] for team in target_player_team_stats)
    total_prob = (total_triumphs / total_picks * 100) if total_picks > 0 else 0
    if total_picks == 0:
        print(f"{'No picks made by player on target date':^{len(headers)}}")
    else:
        for team in sorted(target_player_team_stats.keys()):
            picks = target_player_team_stats[team]['picks']
            # Only show teams that were actually picked
            if picks > 0:
                triumphs = target_player_team_stats[team].get('triumphs', 0)
                pick_pct = (picks / total_picks * 100) if total_picks > 0 else 0
                prob = (triumphs / picks * 100) if picks > 0 else 0
                seed = teams_df[teams_df['teamName'] == team]['seed'].iloc[0] if not teams_df[teams_df['teamName'] == team].empty else None
                team_display = team if seed is None else f"{team} ({seed})"
                print(f"{team_display:<{team_width}} | "
                      f"{picks:>{picks_width}} | "
                      f"{pick_pct:>{pct_width-1}.2f}% | "
                      f"{triumphs:>{triumphs_width}} | " # Add back
                      f"{prob:>{prob_width-1}.2f}%" ) # Add back
    print(separator)
    if total_picks > 0:
         print(f"{'TOTAL':<{team_width}} | "
               f"{total_picks:>{picks_width}} | "
               f"{100:>{pct_width-1}.2f}% | "
               f"{total_triumphs:>{triumphs_width}} | " # Add back
               f"{total_prob:>{prob_width-1}.2f}%" ) # Add back

    # Tournament Winners table
    print("\nTournament Winners:")
    
    # Define column widths
    team_width = 25  # Increased to accommodate seed info
    wins_width = 8
    pct_width = 14
    
    # Create centered headers and separator
    headers = (
        f"{'Team':^{team_width}} | "
        f"{'Wins':^{wins_width}} | "
        f"{'Win Percentage':^{pct_width}}"
    )
    separator = "-" * len(headers)
    
    # Print headers and separator
    print(headers)
    print(separator)
    
    # Print each team's tournament wins
    for team in sorted(tournament_winners.keys()):
        wins = tournament_winners[team]
        percentage = (wins / num_simulations) * 100
        
        # Get team seed
        seed = None
        team_row = teams_df[teams_df['teamName'] == team]
        if not team_row.empty:
            seed = team_row['seed'].iloc[0]
        
        # Format team name with seed if available
        team_display = team if seed is None else f"{team} ({seed})"
        
        print(
            f"{team_display:<{team_width}} | "
            f"{wins:^{wins_width}} | "
            f"{percentage:>{pct_width-1}.2f}%"
        )
    
    # Print totals
    print(separator)
    print(
        f"{'TOTAL':<{team_width}} | "
        f"{num_simulations:^{wins_width}} | "
        f"{100:>{pct_width-1}.2f}%"
    )

def main():
    parser = argparse.ArgumentParser(description='Simulate tournament and survivor pool')
    parser.add_argument('--simulations', type=int, default=20, help='Number of simulations to run')
    parser.add_argument('--target-date', type=str, required=True, help='Target date to analyze (MM/DD/YYYY)')
    parser.add_argument('--target-player', type=int, required=True, help='Target player ID to analyze')
    parser.add_argument('--variance-factor', type=float, default=0, help='How much random variation in picks (0=deterministic, higher values=more random)')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing')
    parser.add_argument('--processes', type=int, help='Number of processes to use (default: number of CPU cores)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Set global debug flag based on command line argument
    global DEBUG
    DEBUG = args.debug
    
    # Run simulations
    run_simulations(
        args.simulations, 
        args.target_date, 
        args.target_player, 
        args.variance_factor,
        use_multiprocessing=not args.no_multiprocessing,
        num_processes=args.processes
    )

if __name__ == "__main__":
    # This is needed for multiprocessing to work correctly on Windows
    multiprocessing.freeze_support()
    main()