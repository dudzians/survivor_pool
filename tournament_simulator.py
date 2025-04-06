import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import random
import csv
import time  # For performance measurement
import os
from functools import lru_cache  # For caching expensive operations
import multiprocessing

# Debug flag - set to False for faster execution, True for debugging
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
    def __init__(self, teams_df, schedule_df, seed_boost_df, variance_factor):
        self.schedule_df = schedule_df
        self.teams_df = teams_df
        self.seed_boost_df = seed_boost_df
        self.variance_factor = variance_factor
        self.winners = {}  # {game_id: winning team}
        self.champion = None  # Store the tournament champion
        
        # Map round numbers to odds column names
        self.round_to_odds = {
            64: '64odds',
            32: '32odds',
            16: '16odds',
            8: '8odds',
            4: '4odds',
            2: '2odds'
        }
        
        # Create cached mappings for faster lookups
        self.team_to_seed = {}
        self.team_to_odds = {}
        
        # Precompute team seed and odds data
        for _, row in self.teams_df.iterrows():
            team_name = row['teamName']
            self.team_to_seed[team_name] = row['seed']
            self.team_to_odds[team_name] = {
                64: row['64odds'] if '64odds' in row else 0,
                32: row['32odds'] if '32odds' in row else 0,
                16: row['16odds'] if '16odds' in row else 0,
                8: row['8odds'] if '8odds' in row else 0,
                4: row['4odds'] if '4odds' in row else 0,
                2: row['2odds'] if '2odds' in row else 0
            }
        
        # Ensure we have a team mapping
        self.team_mapping = create_team_mapping(teams_df, pd.read_csv('sample_picks.csv', index_col='player_id'), schedule_df)
    
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
        # Use cached odds data instead of DataFrame lookup
        if team in self.team_to_odds:
            return self.team_to_odds[team].get(round_num, 0)
        return 0
    
    def simulate_game(self, game_id):
        game = self.schedule_df[self.schedule_df['gameID'] == game_id].iloc[0]
        round_num = game['round']
        team_a = self.get_team_name(game['teamA'])
        team_b = self.get_team_name(game['teamB'])
        
        # Get odds for each team
        odds_a = self.get_team_odds(team_a, round_num)
        odds_b = self.get_team_odds(team_b, round_num)
        
        # If both teams have 0 odds, choose randomly
        if odds_a == 0 and odds_b == 0:
            winner = random.choice([team_a, team_b])
        else:
            # Normalize odds and choose based on probability
            total_odds = odds_a + odds_b
            if total_odds == 0:
                winner = random.choice([team_a, team_b])
            else:
                prob_a = odds_a / total_odds
                if random.random() < prob_a:
                    winner = team_a
                else:
                    winner = team_b
        
        # Record the winner
        self.winners[game_id] = winner
        
        # If this is the final game, set the champion
        if game_id == 'FINAL':
            self.champion = winner
            
        return winner
    
    def get_games_for_date(self, date):
        return self.schedule_df[self.schedule_df['day'] == date]
        
    def simulate_championship(self):
        """Simulate the championship game and determine the tournament champion"""
        # Check if we have already simulated the final game
        if 'FINAL' in self.winners:
            self.champion = self.winners['FINAL']
            return self.champion
            
        try:
            # Find the FINAL game in the schedule
            final_games = self.schedule_df[self.schedule_df['gameID'] == 'FINAL']
            if not final_games.empty:
                self.champion = self.simulate_game('FINAL')
            else:
                # If no FINAL game is found, use the F4 winners to determine champion
                f4_games = self.schedule_df[self.schedule_df['gameID'].str.startswith('F4')]
                
                # Get the winners of the Final Four games
                f4_winners = []
                for _, game in f4_games.iterrows():
                    game_id = game['gameID']
                    if game_id in self.winners:
                        f4_winners.append(self.winners[game_id])
                    else:
                        f4_winners.append(self.simulate_game(game_id))
                
                if len(f4_winners) >= 2:
                    # Simulate a virtual championship game
                    team_a, team_b = f4_winners[0], f4_winners[1]
                    
                    # Get odds for both teams (using round 2 odds)
                    odds_a = self.get_team_odds(team_a, 2)
                    odds_b = self.get_team_odds(team_b, 2)
                    
                    # Normalize odds and choose based on probability
                    total_odds = odds_a + odds_b
                    if total_odds == 0:
                        self.champion = random.choice([team_a, team_b])
                    else:
                        prob_a = odds_a / total_odds
                        if random.random() < prob_a:
                            self.champion = team_a
                        else:
                            self.champion = team_b
                    
                    # Record the winner in the winners dictionary
                    self.winners['FINAL'] = self.champion
                else:
                    # If we can't determine the champion from F4 games, pick a random team
                    self.champion = random.choice(list(self.team_to_seed.keys()))
        except Exception as e:
            if DEBUG:
                print(f"Error simulating championship: {e}")
            # Fallback - pick a random team as champion
            self.champion = random.choice(list(self.team_to_seed.keys()))
            
        return self.champion

class SurvivorPoolSimulator:
    def __init__(self, teams_df, schedule_df, target_date):
        self.schedule_df = schedule_df
        self.teams_df = teams_df
        self.target_date = target_date
        self.variance_factor = 10  # Default variance factor
        
        # Load picks data
        self.picks_df = pd.read_csv('sample_picks.csv', index_col='player_id')
        
        # Create team mapping
        self.team_mapping = create_team_mapping(teams_df, self.picks_df, schedule_df)
        
        # Initialize tracking variables
        self.active_players = set(self.picks_df.index)
        self.eliminated_players = set()
        self.elimination_dates = {}  # {player_id: date eliminated}
        self.used_teams = defaultdict(set)  # {player_id: set of used teams}
        self.target_date_picks = {}  # {player_id: team picked on target date}
        self.target_date_triumphs = set()  # players who triumphed on target date
        self.winner = None
        self.winners = {}  # {game_id: winning team}
        
        # Cache team data for faster lookups
        self.team_to_seed = {}
        for _, row in self.teams_df.iterrows():
            self.team_to_seed[row['teamName']] = row['seed']
        
        # Process predetermined picks
        for player_id in self.active_players:
            for date in self.picks_df.columns:
                pick = self.picks_df.loc[player_id, date]
                if pd.notna(pick):
                    # Map the pick to the official team name
                    pick = self.team_mapping.get(pick, pick)
                    self.used_teams[player_id].add(pick)
    
    def is_complete(self):
        return len(self.active_players - self.eliminated_players) <= 1 or self.winner is not None
    
    def get_player_pick(self, player_id, date):
        if player_id in self.eliminated_players:
            return None
            
        # Check if there's a predetermined pick
        pick = self.picks_df.loc[player_id, date]
        if pd.notna(pick):
            # Map the pick to the official team name
            pick = self.team_mapping.get(pick, pick)
            return pick
            
        # Check if other players have picks for this date
        other_players_have_picks = False
        for other_id in self.active_players:
            if other_id != player_id and other_id not in self.eliminated_players:
                other_pick = self.picks_df.loc[other_id, date]
                if pd.notna(other_pick):
                    other_players_have_picks = True
                    break
        
        # If other players have picks but this player doesn't, eliminate them
        if other_players_have_picks:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            return None
            
        # Get available teams for this player
        available_teams = self.get_available_teams(player_id)
        if not available_teams:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            return None
            
        # Get games for this date
        games = self.schedule_df[self.schedule_df['day'] == date]
        playing_teams = set()
        current_round = None
        
        # Create a lookup dictionary of winners
        winners_lookup = {}
        for game_id, winner in self.winners.items():
            winners_lookup[game_id] = winner
        
        for _, game in games.iterrows():
            # Map team names to official names
            team_a = self.team_mapping.get(game['teamA'], game['teamA'])
            team_b = self.team_mapping.get(game['teamB'], game['teamB'])
            
            # Faster lookup using the winners dictionary
            if team_a.startswith('Winner of:'):
                game_id = team_a.split(':')[1].strip()
                if game_id in winners_lookup:
                    team_a = winners_lookup[game_id]
            if team_b.startswith('Winner of:'):
                game_id = team_b.split(':')[1].strip()
                if game_id in winners_lookup:
                    team_b = winners_lookup[game_id]
                    
            playing_teams.add(team_a)
            playing_teams.add(team_b)
            current_round = game['round']
        
        # Filter available teams to those playing today
        available_teams = [team for team in available_teams if team in playing_teams]
        if not available_teams:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            return None
            
        # Get odds for available teams
        team_odds = {}
        for team in available_teams:
            # Use cached seed data instead of DataFrame lookup
            seed = self.team_to_seed.get(team, 0)
            if seed > 0:  # Only proceed if we found a valid seed
                odds = self.teams_df[self.teams_df['teamName'] == team]['64odds'].iloc[0]  # Use first round odds as baseline
                
                # Convert the current_round from string format to numeric
                numeric_round = current_round
                if isinstance(current_round, str) and current_round.startswith('R'):
                    try:
                        numeric_round = int(current_round[1:])
                    except ValueError:
                        numeric_round = 64  # Default if conversion fails
                
                # Look up the boost value using the numeric round value
                boost_multiplier = SEED_BOOSTS.get(str(numeric_round), {}).get(seed, 1)
                
                # DEBUG print only when DEBUG is True
                if DEBUG:
                    print(f"Round: {current_round}, Team: {team}, Seed: {seed}, Boost: {boost_multiplier}")
                
                # Apply random variance
                variance = random.uniform(-self.variance_factor, self.variance_factor)
                
                # Combine factors to create pick weight with direct boost multiplier
                pick_weight = odds * boost_multiplier * (1 + variance/100)
                team_odds[team] = pick_weight
        
        # Choose team based on pick weights
        if not team_odds:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            return None
            
        total_weight = sum(team_odds.values())
        if total_weight == 0:
            # If all teams have 0 weight, choose randomly
            pick = random.choice(list(team_odds.keys()))
        else:
            # Normalize weights to probabilities
            normalized_weights = {team: weight/total_weight for team, weight in team_odds.items()}
            # Choose team based on normalized weights
            pick = random.choices(list(normalized_weights.keys()), 
                                weights=list(normalized_weights.values()))[0]
        
        # Record the pick
        self.used_teams[player_id].add(pick)
        return pick
    
    def get_available_teams(self, player_id):
        """Get available teams for a player (teams that haven't been used yet)"""
        used = self.used_teams[player_id]
        # Use list comprehension for better performance than filtering DataFrame
        return [team for team in self.team_to_seed.keys() if team not in used]
        
    def get_seed_sum(self, player_id):
        """Calculate sum of seeds for all teams picked by a player"""
        total = 0
        for team in self.used_teams[player_id]:
            # Use cached team seed data instead of DataFrame lookup
            total += self.team_to_seed.get(team, 0)
        return total
        
    def determine_winner(self):
        """Determine the winner based on elimination dates and tiebreakers"""
        if not self.elimination_dates:
            return None
            
        # Group players by elimination date
        players_by_date = defaultdict(list)
        for player_id, date in self.elimination_dates.items():
            players_by_date[date].append(player_id)
            
        # Find the latest date where players were eliminated
        latest_date = max(players_by_date.keys())
        last_players = players_by_date[latest_date]
        
        if len(last_players) == 1:
            return last_players[0]
            
        # First tiebreaker: sum of seeds
        seed_sums = {player: self.get_seed_sum(player) for player in last_players}
        max_sum = max(seed_sums.values())
        players_with_max_sum = [p for p in last_players if seed_sums[p] == max_sum]
        
        if len(players_with_max_sum) == 1:
            return players_with_max_sum[0]
            
        # Second tiebreaker: random selection
        return random.choice(players_with_max_sum)

    def run_simulation(self, tournament_sim):
        # Simulate until pool is complete
        current_date = min(self.schedule_df['day'])
        while not self.is_complete():
            if DEBUG:
                print(f"\nProcessing date: {current_date}")
            
            # Special handling for combined days 7 and 8
            if current_date in ['3/29/2025', '3/30/2025']:
                # Get games for both days
                games_day7 = tournament_sim.get_games_for_date('3/29/2025')
                games_day8 = tournament_sim.get_games_for_date('3/30/2025')
                
                # Combine games from both days
                games = pd.concat([games_day7, games_day8])
                games = games.sort_values('round')
                
                # Track which players are active before any games are played
                active_players_on_date = set()
                player_picks = {}  # {player_id: [pick1, pick2]}
                
                for player_id in self.active_players:
                    if player_id not in self.eliminated_players:
                        # Get picks for both days
                        pick1 = self.get_player_pick(player_id, '3/29/2025')
                        pick2 = self.get_player_pick(player_id, '3/30/2025')
                        
                        if pick1 is not None and pick2 is not None:
                            active_players_on_date.add(player_id)
                            player_picks[player_id] = [pick1, pick2]
                            # Record picks if this is the target date
                            if '3/29/2025' == self.target_date or '3/30/2025' == self.target_date:
                                self.target_date_picks[player_id] = f"{pick1}, {pick2}"
                                if DEBUG:
                                    print(f"Player {player_id} picked {pick1} and {pick2} on target date")
                        else:
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            if DEBUG:
                                print(f"Player {player_id} eliminated (no picks available)")
                
                if DEBUG:
                    print(f"Active players on combined days: {len(active_players_on_date)}")
                    print(f"Eliminated players: {len(self.eliminated_players)}")
                
                # Simulate each game
                for _, game in games.iterrows():
                    if DEBUG:
                        print(f"\nSimulating game: {game['gameID']}")
                    # Simulate game
                    winner = tournament_sim.simulate_game(game['gameID'])
                    self.winners[game['gameID']] = winner
                    if DEBUG:
                        print(f"Winner: {winner}")
                    
                    # Process each active player's picks
                    for player_id in list(active_players_on_date):
                        if player_id in self.eliminated_players:
                            continue
                            
                        picks = player_picks.get(player_id)
                        if picks is None:
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            if DEBUG:
                                print(f"Player {player_id} eliminated (no picks available)")
                            continue
                            
                        # Get the teams playing in this game
                        team_a = tournament_sim.get_team_name(game['teamA'])
                        team_b = tournament_sim.get_team_name(game['teamB'])
                        
                        # Check if either of the player's picks lost in this game
                        if (picks[0] in [team_a, team_b] and picks[0] != winner) or \
                           (picks[1] in [team_a, team_b] and picks[1] != winner):
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            if DEBUG:
                                print(f"Player {player_id} eliminated (picked {picks[0]} and {picks[1]}, winner was {winner})")
                        elif picks[0] == winner or picks[1] == winner:
                            if DEBUG:
                                print(f"Player {player_id} survived with picks {picks[0]} and {picks[1]}")
                
                # Move to next date after the combined days
                current_date = get_next_date('3/30/2025', self.schedule_df)
                continue
            
            # Normal processing for other days
            games = tournament_sim.get_games_for_date(current_date)
            games = games.sort_values('round')
            
            # Track which players are active before any games are played
            active_players_on_date = set()
            player_picks = {}  # {player_id: pick}
            
            for player_id in self.active_players:
                if player_id not in self.eliminated_players:
                    pick = self.get_player_pick(player_id, current_date)
                    if pick is not None:
                        active_players_on_date.add(player_id)
                        player_picks[player_id] = pick
                        # Record pick if this is the target date
                        if current_date == self.target_date:
                            self.target_date_picks[player_id] = pick
                            if DEBUG:
                                print(f"Player {player_id} picked {pick} on target date")
                    else:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        if DEBUG:
                            print(f"Player {player_id} eliminated (no pick available)")
            
            if DEBUG:
                print(f"Active players on {current_date}: {len(active_players_on_date)}")
                print(f"Eliminated players: {len(self.eliminated_players)}")
            
            # Simulate each game
            for _, game in games.iterrows():
                if DEBUG:
                    print(f"\nSimulating game: {game['gameID']}")
                # Simulate game
                winner = tournament_sim.simulate_game(game['gameID'])
                self.winners[game['gameID']] = winner
                if DEBUG:
                    print(f"Winner: {winner}")
                
                # Process each active player's pick
                for player_id in list(active_players_on_date):
                    if player_id in self.eliminated_players:
                        continue
                        
                    pick = player_picks.get(player_id)
                    if pick is None:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        if DEBUG:
                            print(f"Player {player_id} eliminated (no pick available)")
                        continue
                        
                    # Get the teams playing in this game
                    team_a = tournament_sim.get_team_name(game['teamA'])
                    team_b = tournament_sim.get_team_name(game['teamB'])
                    
                    # Only eliminate player if their picked team lost in this game
                    if pick in [team_a, team_b] and pick != winner:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        if DEBUG:
                            print(f"Player {player_id} eliminated (picked {pick}, winner was {winner})")
                    elif pick == winner and current_date == self.target_date:
                        if DEBUG:
                            print(f"Player {player_id} survived with pick {pick}")
            
            # Check if we have a winner after all games for the day
            active_count = len(self.active_players - self.eliminated_players)
            if DEBUG:
                print(f"\nAfter processing all games for {current_date}:")
                print(f"Active players: {active_count}")
                print(f"Eliminated players: {len(self.eliminated_players)}")
            
            if active_count == 0:
                # All players have been eliminated, determine winner based on who lasted longest
                self.winner = self.determine_winner()
                if DEBUG:
                    print(f"Pool complete! Winner: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            elif active_count == 1:
                # One player still active, they win
                self.winner = list(self.active_players - self.eliminated_players)[0]
                if DEBUG:
                    print(f"Pool complete! Winner: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            
            current_date = get_next_date(current_date, self.schedule_df)
            if current_date is None:
                # Reached end of schedule, determine winner based on who lasted longest
                self.winner = self.determine_winner()
                if DEBUG:
                    print(f"Reached end of schedule. Winner: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break

def get_next_date(current_date, schedule_df):
    """Get the next date in the schedule after current_date."""
    dates = sorted(schedule_df['day'].unique())
    current_idx = dates.index(current_date)
    if current_idx + 1 < len(dates):
        return dates[current_idx + 1]
    return None

def run_simulations(num_simulations, target_player, target_date, variance_factor=10, show_progress=True):
    start_time = time.time()
    
    # Load data once outside simulation loop
    schedule_df, teams_df, picks_df, seed_boost_df = load_data()
    
    # Tracking variables
    player_stats = defaultdict(lambda: {"picks": 0, "triumphs": 0})
    team_stats = defaultdict(lambda: {"picks": 0, "triumphs": 0})
    target_player_team_stats = defaultdict(lambda: {"picks": 0, "triumphs": 0})
    tournament_winners = defaultdict(int)
    
    # Check if we should use multiprocessing
    use_mp = num_simulations > 4 and multiprocessing.cpu_count() > 1
    
    if use_mp:
        try:
            # Define a worker function for multiprocessing
            def run_single_simulation(i):
                # Set a unique random seed for each process
                np.random.seed(int(time.time() * 1000000) % (2**32 - 1) + i)
                
                # Create tournament simulator and survivor pool simulator
                tournament_sim = TournamentSimulator(teams_df, schedule_df, seed_boost_df, variance_factor)
                pool_sim = SurvivorPoolSimulator(teams_df, schedule_df, target_date)
                
                # Run the simulation
                pool_sim.run_simulation(tournament_sim)
                
                # Make sure the championship game is simulated and a champion is set
                tournament_sim.simulate_championship()
                
                # Collect results
                results = {
                    "tournament_winner": tournament_sim.champion,
                    "pool_winner": pool_sim.winner,
                    "target_date_picks": dict(pool_sim.target_date_picks),
                    "target_date_triumphs": list(pool_sim.target_date_triumphs)
                }
                return results
            
            # Create process pool
            with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), num_simulations)) as pool:
                # Map simulation function to process pool
                results = list(pool.map(run_single_simulation, range(num_simulations)))
                
                # Process results
                for result in results:
                    # Track tournament winners
                    tournament_winners[result["tournament_winner"]] += 1
                    
                    # Track player picks and triumphs
                    for player_id, pick in result["target_date_picks"].items():
                        player_stats[player_id]["picks"] += 1
                        if player_id in result["target_date_triumphs"]:
                            player_stats[player_id]["triumphs"] += 1
                        
                        # Track team stats for all players
                        team_stats[pick]["picks"] += 1
                        if player_id in result["target_date_triumphs"]:
                            team_stats[pick]["triumphs"] += 1
                        
                        # Track team stats for target player
                        if player_id == target_player:
                            target_player_team_stats[pick]["picks"] += 1
                            if player_id in result["target_date_triumphs"]:
                                target_player_team_stats[pick]["triumphs"] += 1
        
        except Exception as e:
            if DEBUG:
                print(f"Multiprocessing error: {e}")
                print("Falling back to sequential processing")
            use_mp = False
    
    # Sequential processing if multiprocessing fails or is not used
    if not use_mp:
        for i in range(num_simulations):
            if show_progress and i % 10 == 0:
                print(f"Running simulation {i+1}/{num_simulations}")
            
            # Create tournament simulator and survivor pool simulator
            tournament_sim = TournamentSimulator(teams_df, schedule_df, seed_boost_df, variance_factor)
            pool_sim = SurvivorPoolSimulator(teams_df, schedule_df, target_date)
            
            # Run the simulation
            pool_sim.run_simulation(tournament_sim)
            
            # Make sure the championship game is simulated and a champion is set
            tournament_sim.simulate_championship()
            
            # Track tournament winners
            tournament_winners[tournament_sim.champion] += 1
            
            # Track player picks and triumphs
            for player_id, pick in pool_sim.target_date_picks.items():
                player_stats[player_id]["picks"] += 1
                if player_id in pool_sim.target_date_triumphs:
                    player_stats[player_id]["triumphs"] += 1
                
                # Track team stats for all players
                team_stats[pick]["picks"] += 1
                if player_id in pool_sim.target_date_triumphs:
                    team_stats[pick]["triumphs"] += 1
                
                # Track team stats for target player
                if player_id == target_player:
                    target_player_team_stats[pick]["picks"] += 1
                    if player_id in pool_sim.target_date_triumphs:
                        target_player_team_stats[pick]["triumphs"] += 1
    
    # Calculate statistics
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Output player statistics
    print("\nPlayer Statistics:")
    print("Player ID | Picks on Target Date | Total Triumphs | Triumph Probability")
    print("---------------------------------------------------------------------------")
    total_picks = 0
    total_triumphs = 0
    for player_id in sorted(player_stats.keys()):
        picks = player_stats[player_id]["picks"]
        triumphs = player_stats[player_id]["triumphs"]
        total_picks += picks
        total_triumphs += triumphs
        triumph_pct = (triumphs / picks * 100) if picks > 0 else 0
        print(f"{player_id:9} |{picks:20} |{triumphs:14} |{triumph_pct:9.2f}%")
    
    if total_picks > 0:
        # Print total row
        total_triumph_pct = (total_triumphs / total_picks * 100) if total_picks > 0 else 0
        print("---------------------------------------------------------------------------")
        print(f"TOTAL     |{total_picks:20} |{total_triumphs:14} |{total_triumph_pct:9.2f}%")
    
    # Output team statistics for all players
    print("\nTeam Statistics (All Players):")
    print("Team | Picks on Target Date | Triumphs | Triumph Probability")
    print("------------------------------------------------------------")
    total_team_picks = 0
    total_team_triumphs = 0
    for team in sorted(team_stats.keys()):
        picks = team_stats[team]["picks"]
        triumphs = team_stats[team]["triumphs"]
        total_team_picks += picks
        total_team_triumphs += triumphs
        triumph_pct = (triumphs / picks * 100) if picks > 0 else 0
        print(f"{team:20} |{picks:20} |{triumphs:9} |{triumph_pct:9.2f}%")
    
    if total_team_picks > 0:
        # Print total row
        total_team_triumph_pct = (total_team_triumphs / total_team_picks * 100) if total_team_picks > 0 else 0
        print("------------------------------------------------------------")
        print(f"TOTAL                |{total_team_picks:20} |{total_team_triumphs:9} |{total_team_triumph_pct:9.2f}%")
    
    # Output team statistics for target player
    print(f"\nTeam Statistics (Player {target_player}):")
    print("Team | Picks on Target Date | Triumphs | Triumph Probability")
    print("------------------------------------------------------------")
    total_target_picks = 0
    total_target_triumphs = 0
    for team in sorted(target_player_team_stats.keys()):
        picks = target_player_team_stats[team]["picks"]
        triumphs = target_player_team_stats[team]["triumphs"]
        total_target_picks += picks
        total_target_triumphs += triumphs
        triumph_pct = (triumphs / picks * 100) if picks > 0 else 0
        print(f"{team:20} |{picks:20} |{triumphs:9} |{triumph_pct:9.2f}%")
    
    if total_target_picks > 0:
        # Print total row
        total_target_triumph_pct = (total_target_triumphs / total_target_picks * 100) if total_target_picks > 0 else 0
        print("------------------------------------------------------------")
        print(f"TOTAL                |{total_target_picks:20} |{total_target_triumphs:9} |{total_target_triumph_pct:9.2f}%")
    
    # Output tournament winner statistics
    print("\nTournament Winners:")
    print("Team | Wins | Win Percentage")
    print("----------------------------------------")
    for team, wins in sorted(tournament_winners.items(), key=lambda x: x[1], reverse=True):
        if team is None:
            team_name = "Unknown"
        else:
            team_name = str(team)
        win_pct = wins / num_simulations * 100
        print(f"{team_name:20} |{wins:5} |{win_pct:8.2f}%")
    
    # Print total row
    print("----------------------------------------")
    print(f"TOTAL                |{num_simulations:5} |{100:8.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Simulate tournament and survivor pool')
    parser.add_argument('--simulations', type=int, default=20, help='Number of simulations to run')
    parser.add_argument('--target-date', type=str, required=True, help='Target date to analyze (MM/DD/YYYY)')
    parser.add_argument('--target-player', type=int, required=True, help='Target player ID to analyze')
    parser.add_argument('--variance-factor', type=int, default=5, 
                        help='How much random variation in picks (1-10). Higher values = more randomness.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    args = parser.parse_args()

    # Set global debug flag based on command line argument
    global DEBUG
    DEBUG = args.debug
    
    # Read input files
    try:
        schedule_df = pd.read_csv('schedule.csv')
        teams_df = pd.read_csv('teams.csv')
        picks_df = pd.read_csv('sample_picks.csv')
        
        # Set player_id as index for picks_df
        picks_df.set_index('player_id', inplace=True)
        
        # Print picks DataFrame info if debug mode is enabled
        if DEBUG:
            print("\nPicks DataFrame Info:")
            print(f"Columns: {picks_df.columns.tolist()}")
            print(f"Index: {picks_df.index.tolist()[:5]}...")
            print(f"Sample picks for target date {args.target_date}:")
            print(picks_df[args.target_date].head())
        
        # Create team mapping
        team_mapping = create_team_mapping(teams_df, picks_df, schedule_df)
        
        if DEBUG:
            print("\nTeam mapping summary:")
            for team, mapped in sorted(team_mapping.items()):
                print(f"{team} -> {mapped}")
        
        # Run simulations
        run_simulations(args.simulations, args.target_player, args.target_date, args.variance_factor)
    except Exception as e:
        print(f"Error during execution: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())