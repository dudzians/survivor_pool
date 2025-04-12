import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import random
import csv
import multiprocessing
from functools import partial

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
            print(f"DEBUG get_team_odds: Team '{team}' not found in teams_df.") # DEBUG
            return 0
        # Ensure round_num is an integer key if it's not already
        try:
            round_key = int(round_num) 
        except ValueError:
             print(f"DEBUG get_team_odds: Invalid round_num '{round_num}' for team '{team}'. Defaulting key to 64.") # DEBUG
             round_key = 64 # Or handle error appropriately

        if round_key not in self.round_to_odds:
             print(f"DEBUG get_team_odds: Round key '{round_key}' not in round_to_odds mapping for team '{team}'. Defaulting to 64odds.") # DEBUG
             odds_col = '64odds'
        else:
             odds_col = self.round_to_odds[round_key]

        odds_value = team_row[odds_col].iloc[0]
        print(f"DEBUG get_team_odds: Team='{team}', Round={round_num}, OddsColumn='{odds_col}', FetchedOdds={odds_value}") # DEBUG
        return odds_value
    
    def simulate_game(self, game_id):
        game = self.schedule_df[self.schedule_df['gameID'] == game_id].iloc[0]
        round_num = game['round']
        team_a_raw = game['teamA']
        team_b_raw = game['teamB']
        print(f"\nDEBUG simulate_game: Simulating GameID='{game_id}', Round={round_num}, RawTeams=['{team_a_raw}', '{team_b_raw}']") # DEBUG
        
        # Resolve team names (this might recursively call simulate_game)
        team_a = self.get_team_name(team_a_raw)
        team_b = self.get_team_name(team_b_raw)
        print(f"DEBUG simulate_game: ResolvedTeams=['{team_a}', '{team_b}']") # DEBUG
        
        # Get odds for each team for the correct round
        odds_a = self.get_team_odds(team_a, round_num)
        odds_b = self.get_team_odds(team_b, round_num)
        print(f"DEBUG simulate_game: Odds=['{team_a}': {odds_a}, '{team_b}': {odds_b}]") # DEBUG
        
        # Determine winner based on odds (should be deterministic for 100/0)
        winner = None
        if odds_a == 0 and odds_b == 0:
            print("DEBUG simulate_game: Both teams have 0 odds! This shouldn't happen with 100/0 input. Choosing randomly.") # DEBUG
            winner = random.choice([team_a, team_b])
        elif odds_a > 0 and odds_b == 0:
             winner = team_a
        elif odds_b > 0 and odds_a == 0:
             winner = team_b
        elif odds_a > 0 and odds_b > 0: # Both have > 0 odds? Should only be if both are 100?
             print("DEBUG simulate_game: Both teams have > 0 odds! Applying probability.") # DEBUG
             # Normalize odds and choose based on probability
             total_odds = odds_a + odds_b
             if total_odds == 0: # Should not happen if odds > 0
                 winner = random.choice([team_a, team_b])
             else:
                 prob_a = odds_a / total_odds
                 rand_val = random.random() # Generate random number
                 print(f"DEBUG simulate_game: ProbA={prob_a:.4f}, RandVal={rand_val:.4f}") # DEBUG
                 if rand_val < prob_a:
                     winner = team_a
                 else:
                     winner = team_b
        else: # Should not happen if one is 100 and other is 0
             print(f"DEBUG simulate_game: Unexpected odds combo! odds_a={odds_a}, odds_b={odds_b}. Choosing randomly.") # DEBUG
             winner = random.choice([team_a, team_b])

        # Record the winner
        self.winners[game_id] = winner
        print(f"DEBUG simulate_game: Determined Winner for '{game_id}': {winner}") # DEBUG
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
            for date in picks_df.columns:
                pick = picks_df.loc[player_id, date]
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
        # Use .get() with a default for the date column to avoid KeyError if column missing
        pick = self.picks_df.get(date, pd.Series(index=self.picks_df.index)).loc[player_id]

        if pd.notna(pick):
            original_pick = pick
            pick = self.team_mapping.get(pick, pick)
            if pick == '':
                return None # Treat predetermined empty pick as None
            
            # Record the predetermined pick as used *if it hasn't been added already* 
            # (The init loop handles adding picks, this ensures it if logic changes)
            if pick not in self.used_teams[player_id]:
                 self.used_teams[player_id].add(pick) 
            return pick

        # Check if other players have picks for this date (elimination rule)
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
        for _, game in games.iterrows():
            # Map team names to official names
            team_a = self.team_mapping.get(game['teamA'], game['teamA'])
            team_b = self.team_mapping.get(game['teamB'], game['teamB'])
            if team_a.startswith('Winner of:'):
                game_id = team_a.split(':')[1].strip()
                if game_id in self.winners:
                    team_a = self.winners[game_id]
            if team_b.startswith('Winner of:'):
                game_id = team_b.split(':')[1].strip()
                if game_id in self.winners:
                    team_b = self.winners[game_id]
            playing_teams.add(team_a)
            playing_teams.add(team_b)
            current_round = game['round'] # Capture round here

        # Filter available teams to those playing today
        available_teams_playing = [team for team in available_teams if team in playing_teams]
        
        if not available_teams_playing:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            return None
            
        # Get odds for available teams
        team_odds = {}
        for team in available_teams_playing:
            team_row = self.teams_df[self.teams_df['teamName'] == team]
            if not team_row.empty:
                seed = team_row['seed'].iloc[0]
                # Use the captured current_round for odds calculation
                odds = self.get_team_odds_for_pick(team, current_round) # Changed to use specific method if needed, or ensure get_team_odds handles round correctly
                boost_multiplier = SEED_BOOSTS.get(str(current_round), {}).get(seed, 1) # Assuming current_round matches SEED_BOOSTS keys
                pick_weight = odds * boost_multiplier # Simplified weight for deterministic check
                team_odds[team] = pick_weight
        
        # Choose team based on pick weights (should be deterministic if weights differ)
        if not team_odds:
            self.eliminated_players.add(player_id)
            self.elimination_dates[player_id] = date
            return None
            
        # Simplified choice for deterministic: pick team with max weight
        # If ties, default to sorting alphabetically to ensure determinism
        max_weight = -1
        best_teams = []
        for team, weight in team_odds.items():
            if weight > max_weight:
                max_weight = weight
                best_teams = [team]
            elif weight == max_weight:
                best_teams.append(team)
                
        if not best_teams: # Should not happen if team_odds is not empty
             self.eliminated_players.add(player_id)
             self.elimination_dates[player_id] = date
             return None
             
        best_teams.sort() # Deterministic tie-break
        pick = best_teams[0]
        
        # Record the pick
        self.used_teams[player_id].add(pick)
        return pick

    def get_team_odds_for_pick(self, team, round_num):
        # Wrapper or specific implementation for getting odds based on current round for picks
        # This should use the logic similar to get_team_odds but ensure round_num is correct
        # For now, assume get_team_odds handles the round correctly based on previous fixes
        return self.get_team_odds(team, round_num) 

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
                        else:
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                
                # Simulate each game
                for _, game in games.iterrows():
                    # Simulate game
                    winner = tournament_sim.simulate_game(game['gameID'])
                    self.winners[game['gameID']] = winner
                    
                    # Process each active player's picks
                    for player_id in list(active_players_on_date):
                        if player_id in self.eliminated_players:
                            continue
                            
                        picks = player_picks.get(player_id)
                        if picks is None:
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                            continue
                            
                        # Get the teams playing in this game
                        team_a = tournament_sim.get_team_name(game['teamA'])
                        team_b = tournament_sim.get_team_name(game['teamB'])
                        
                        # Check if either of the player's picks lost in this game
                        if (picks[0] in [team_a, team_b] and picks[0] != winner) or \
                           (picks[1] in [team_a, team_b] and picks[1] != winner):
                            self.eliminated_players.add(player_id)
                            self.elimination_dates[player_id] = current_date
                
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
                    else:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
            
            # Simulate each game
            for _, game in games.iterrows():
                # Simulate game
                winner = tournament_sim.simulate_game(game['gameID'])
                self.winners[game['gameID']] = winner
                
                # Process each active player's pick
                for player_id in list(active_players_on_date):
                    if player_id in self.eliminated_players:
                        continue
                        
                    pick = player_picks.get(player_id)
                    if pick is None:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        continue
                        
                    # Get the teams playing in this game
                    team_a = tournament_sim.get_team_name(game['teamA'])
                    team_b = tournament_sim.get_team_name(game['teamB'])
                    
                    # Only eliminate player if their picked team lost in this game
                    if pick in [team_a, team_b] and pick != winner:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
            
            # Check if we have a winner after all games for the day
            active_count = len(self.active_players - self.eliminated_players)
            
            if active_count == 0:
                # All players have been eliminated, determine winner based on who lasted longest
                self.winner = self.determine_winner()
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            elif active_count == 1:
                # One player still active, they win
                self.winner = list(self.active_players - self.eliminated_players)[0]
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            
            current_date = get_next_date(current_date, self.schedule_df)
            if current_date is None:
                # Reached end of schedule, determine winner based on who lasted longest
                self.winner = self.determine_winner()
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

def run_single_simulation(sim_number, schedule_df, teams_df, picks_df, team_mapping, target_date, target_player, variance_factor):
    """Run a single simulation and return the results"""
    # We don't print the simulation number here anymore if using multiprocessing
    # print(f"Running simulation {sim_number+1}") 
    
    # Create new simulators for this simulation
    tournament_sim = TournamentSimulator(schedule_df.copy(), teams_df.copy(), team_mapping.copy()) # Use copies to ensure isolation
    survivor_sim = SurvivorPoolSimulator(schedule_df.copy(), teams_df.copy(), picks_df.copy(), variance_factor, target_date) # Use copies
    
    # Run the simulation
    survivor_sim.run_simulation(tournament_sim)
    
    # Ensure we have a winner set
    if survivor_sim.winner is None:
        survivor_sim.winner = survivor_sim.determine_winner()
        if survivor_sim.winner is not None and survivor_sim.winner in survivor_sim.target_date_picks:
            survivor_sim.target_date_triumphs.add(survivor_sim.winner)
    
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
        'player_triumphs': survivor_sim.target_date_triumphs.copy(),
        'target_player_pick': survivor_sim.target_date_picks.get(target_player)
    }
        
    return results

def run_simulations(num_simulations, target_date, target_player, variance_factor, use_multiprocessing=True, num_processes=None):
    # Load data once
    schedule_df, teams_df, picks_df, team_mapping = load_data()
    
    # Initialize counters for statistics
    player_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0})
    team_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0})
    target_player_team_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0})
    tournament_winners = defaultdict(int)
    all_tournament_outcomes = [] # Store full outcomes for comparison

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
                print(f"Processing result {processed_count}/{num_simulations}", end='\\r') # Progress indicator
                
                # Aggregate results
                if sim_result['tournament_winner']:
                     tournament_winners[sim_result['tournament_winner']] += 1
                all_tournament_outcomes.append(sim_result['full_tournament_outcome'])
                
                # Process player picks and triumphs (same logic as before)
                for player_id, pick in sim_result['player_picks'].items():
                    player_stats[player_id]['picks'] += 1
                    # Check if pick is a string before splitting
                    if isinstance(pick, str) and ',' in pick:
                        picks_list = [p.strip() for p in pick.split(',')]
                        for team in picks_list:
                            # Ensure team name is valid before using as key
                            if team:
                                team_stats[team]['picks'] += 1
                    elif pick: # Ensure pick is not None or empty
                        team_stats[pick]['picks'] += 1
                    
                    if player_id in sim_result['player_triumphs']:
                        player_stats[player_id]['triumphs'] += 1
                        # Check if pick is a string before splitting
                        if isinstance(pick, str) and ',' in pick:
                            picks_list = [p.strip() for p in pick.split(',')]
                            for team in picks_list:
                                # Ensure team name is valid
                                if team:
                                    team_stats[team]['triumphs'] += 1
                        elif pick: # Ensure pick is not None or empty
                             team_stats[pick]['triumphs'] += 1
                        
                        if player_id == target_player:
                            # Check if pick is a string before splitting
                            if isinstance(pick, str) and ',' in pick:
                                picks_list = [p.strip() for p in pick.split(',')]
                                for team in picks_list:
                                     # Ensure team name is valid
                                     if team:
                                         target_player_team_stats[team]['triumphs'] += 1
                            elif pick: # Ensure pick is not None or empty
                                 target_player_team_stats[pick]['triumphs'] += 1

                # Process target player's pick (same logic as before)
                if sim_result['target_player_pick']:
                    pick = sim_result['target_player_pick']
                    # Check if pick is a string before splitting
                    if isinstance(pick, str) and ',' in pick:
                        picks_list = [p.strip() for p in pick.split(',')]
                        for team in picks_list:
                             # Ensure team name is valid
                             if team:
                                 target_player_team_stats[team]['picks'] += 1
                    elif pick: # Ensure pick is not None or empty
                         target_player_team_stats[pick]['picks'] += 1

            print("\\nResults processing complete.") # Newline after progress indicator
            
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

            # Aggregate results
            if sim_result['tournament_winner']:
                 tournament_winners[sim_result['tournament_winner']] += 1
            all_tournament_outcomes.append(sim_result['full_tournament_outcome'])

            # Process player picks and triumphs (same logic as before)
            for player_id, pick in sim_result['player_picks'].items():
                 player_stats[player_id]['picks'] += 1
                 # Check if pick is a string before splitting
                 if isinstance(pick, str) and ',' in pick:
                     picks_list = [p.strip() for p in pick.split(',')]
                     for team in picks_list:
                         # Ensure team name is valid
                         if team:
                             team_stats[team]['picks'] += 1
                 elif pick:
                     team_stats[pick]['picks'] += 1
                 
                 if player_id in sim_result['player_triumphs']:
                     player_stats[player_id]['triumphs'] += 1
                     # Check if pick is a string before splitting
                     if isinstance(pick, str) and ',' in pick:
                         picks_list = [p.strip() for p in pick.split(',')]
                         for team in picks_list:
                             # Ensure team name is valid
                             if team:
                                 team_stats[team]['triumphs'] += 1
                     elif pick:
                          team_stats[pick]['triumphs'] += 1
                     
                     if player_id == target_player:
                         # Check if pick is a string before splitting
                         if isinstance(pick, str) and ',' in pick:
                             picks_list = [p.strip() for p in pick.split(',')]
                             for team in picks_list:
                                  # Ensure team name is valid
                                  if team:
                                      target_player_team_stats[team]['triumphs'] += 1
                         elif pick:
                              target_player_team_stats[pick]['triumphs'] += 1

            # Process target player's pick (same logic as before)
            if sim_result['target_player_pick']:
                 pick = sim_result['target_player_pick']
                 # Check if pick is a string before splitting
                 if isinstance(pick, str) and ',' in pick:
                     picks_list = [p.strip() for p in pick.split(',')]
                     for team in picks_list:
                          # Ensure team name is valid
                          if team:
                              target_player_team_stats[team]['picks'] += 1
                 elif pick:
                      target_player_team_stats[pick]['picks'] += 1

    # --- Determinism Check ---
    print("\\n--- Tournament Determinism Check ---")
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

    # After all simulations, print statistics with teams_df
    print_statistics(num_simulations, player_stats, team_stats, target_player_team_stats, tournament_winners, target_player, teams_df)

def print_statistics(num_simulations, player_stats, team_stats, target_player_team_stats, tournament_winners, target_player, teams_df):
    """Print all statistics tables with formatting and team seeds."""
    # Player Statistics
    print("\nPlayer Statistics:")
    print("Player ID | Picks on Target Date | Total Triumphs | Triumph Probability")
    print("-" * 75)
    for player_id in sorted(player_stats.keys()):
        stats = player_stats[player_id]
        prob = (stats['triumphs'] / stats['picks'] * 100) if stats['picks'] > 0 else 0
        print(f"{player_id:9d} | {stats['picks']:20d} | {stats['triumphs']:14d} | {prob:8.2f}%")
    print("-" * 75)
    total_picks = sum(s['picks'] for s in player_stats.values())
    total_triumphs = sum(s['triumphs'] for s in player_stats.values())
    total_prob = (total_triumphs / total_picks * 100) if total_picks > 0 else 0
    print(f"TOTAL     | {total_picks:20d} | {total_triumphs:14d} | {total_prob:8.2f}%")
    
    # Team Statistics (All Players)
    print("\nTeam Statistics (All Players):")
    
    # Define column widths
    team_width = 25  # Increased to accommodate seed info
    picks_width = 20
    pct_width = 12
    triumphs_width = 10
    prob_width = 12
    
    # Create centered headers and separator line
    headers = (
        f"{'Team':^{team_width}} | "
        f"{'Picks on Target Date':^{picks_width}} | "
        f"{'% of Picks':^{pct_width}} | "
        f"{'Triumphs':^{triumphs_width}} | "
        f"{'Triumph Probability':^{prob_width}}"
    )
    separator = "-" * len(headers)
    
    # Print headers and separator
    print(headers)
    print(separator)
    
    # Calculate totals
    total_picks = sum(team_stats[team]['picks'] for team in team_stats)
    total_triumphs = sum(team_stats[team]['triumphs'] for team in team_stats)
    total_prob = (total_triumphs / total_picks * 100) if total_picks > 0 else 0
    
    # Print each team's stats
    for team in sorted(team_stats.keys()):
        picks = team_stats[team]['picks']
        triumphs = team_stats[team]['triumphs']
        pick_pct = (picks / total_picks * 100) if total_picks > 0 else 0
        prob = (triumphs / picks * 100) if picks > 0 else 0
        
        # Get team seed
        seed = None
        team_row = teams_df[teams_df['teamName'] == team]
        if not team_row.empty:
            seed = team_row['seed'].iloc[0]
        
        # Format team name with seed if available
        team_display = team if seed is None else f"{team} ({seed})"
        
        print(
            f"{team_display:<{team_width}} | "
            f"{picks:>{picks_width}} | "
            f"{pick_pct:>{pct_width-1}.2f}% | "
            f"{triumphs:>{triumphs_width}} | "
            f"{prob:>{prob_width-1}.2f}%"
        )
    
    # Print totals
    print(separator)
    print(
        f"{'TOTAL':<{team_width}} | "
        f"{total_picks:>{picks_width}} | "
        f"{100:>{pct_width-1}.2f}% | "
        f"{total_triumphs:>{triumphs_width}} | "
        f"{total_prob:>{prob_width-1}.2f}%"
    )
    
    # Team Statistics (Target Player)
    print(f"\nTeam Statistics (Player {target_player}):")
    
    # Print headers and separator again (reusing same format)
    print(headers)
    print(separator)
    
    # Calculate totals for target player
    total_picks = sum(target_player_team_stats[team]['picks'] for team in target_player_team_stats)
    total_triumphs = sum(target_player_team_stats[team]['triumphs'] for team in target_player_team_stats)
    total_prob = (total_triumphs / total_picks * 100) if total_picks > 0 else 0
    
    # Print each team's stats for target player
    for team in sorted(target_player_team_stats.keys()):
        picks = target_player_team_stats[team]['picks']
        triumphs = target_player_team_stats[team]['triumphs']
        pick_pct = (picks / total_picks * 100) if total_picks > 0 else 0
        prob = (triumphs / picks * 100) if picks > 0 else 0
        
        # Get team seed
        seed = None
        team_row = teams_df[teams_df['teamName'] == team]
        if not team_row.empty:
            seed = team_row['seed'].iloc[0]
        
        # Format team name with seed if available
        team_display = team if seed is None else f"{team} ({seed})"
        
        print(
            f"{team_display:<{team_width}} | "
            f"{picks:>{picks_width}} | "
            f"{pick_pct:>{pct_width-1}.2f}% | "
            f"{triumphs:>{triumphs_width}} | "
            f"{prob:>{prob_width-1}.2f}%"
        )
    
    # Print totals
    print(separator)
    print(
        f"{'TOTAL':<{team_width}} | "
        f"{total_picks:>{picks_width}} | "
        f"{100:>{pct_width-1}.2f}% | "
        f"{total_triumphs:>{triumphs_width}} | "
        f"{total_prob:>{prob_width-1}.2f}%"
    )
    
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
    parser.add_argument('--variance-factor', type=int, default=5, help='How much random variation in picks (1-10)')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing')
    parser.add_argument('--processes', type=int, help='Number of processes to use (default: number of CPU cores)')
    args = parser.parse_args()

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