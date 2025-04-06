import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import random

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
        odds_col = self.round_to_odds[round_num]
        return team_row[odds_col].iloc[0]
    
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
        return winner
    
    def get_games_for_date(self, date):
        return self.schedule_df[self.schedule_df['day'] == date]

class SurvivorPoolSimulator:
    def __init__(self, schedule_df, teams_df, picks_df, favorites_factor, variance_factor, target_date):
        self.schedule_df = schedule_df
        self.teams_df = teams_df
        self.picks_df = picks_df
        self.favorites_factor = favorites_factor
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
            team_row = self.teams_df[self.teams_df['teamName'] == team]
            if not team_row.empty:
                seed = team_row['seed'].iloc[0]
                odds = team_row['64odds'].iloc[0]  # Use first round odds as baseline
                
                # Adjust favorites factor based on round
                if current_round in [64, 32]:
                    # Favor seeds 4-6 in early rounds
                    if 4 <= seed <= 6:
                        seed_factor = (17 - seed) * (self.favorites_factor * 5.0)  # Much stronger boost for seeds 4-6
                    else:
                        seed_factor = (17 - seed) * self.favorites_factor
                elif current_round in [16, 8]:
                    # Favor seeds 2-4 in middle rounds
                    if 2 <= seed <= 4:
                        seed_factor = (17 - seed) * (self.favorites_factor * 5.0)  # Much stronger boost for seeds 2-4
                    else:
                        seed_factor = (17 - seed) * self.favorites_factor
                else:  # R4 or Final
                    # Favor 1-seeds in late rounds
                    if seed == 1:
                        seed_factor = (17 - seed) * (self.favorites_factor * 5.0)  # Much stronger boost for 1-seeds
                    else:
                        seed_factor = (17 - seed) * self.favorites_factor
                
                # Apply random variance
                variance = random.uniform(-self.variance_factor, self.variance_factor)
                # Combine factors to create pick weight
                pick_weight = odds * (1 + seed_factor/100) * (1 + variance/100)
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
            print(f"\nProcessing date: {current_date}")
            # Get games for current date
            games = tournament_sim.get_games_for_date(current_date)
            
            # Sort games by round to ensure we simulate earlier rounds first
            games = games.sort_values('round')
            
            # Track which players are active on this date before any games are played
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
                            print(f"Player {player_id} picked {pick} on target date")
                    else:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        print(f"Player {player_id} eliminated (no pick available)")
            
            print(f"Active players on {current_date}: {len(active_players_on_date)}")
            print(f"Eliminated players: {len(self.eliminated_players)}")
            
            # Simulate each game
            for _, game in games.iterrows():
                print(f"\nSimulating game: {game['gameID']}")
                # Simulate game
                winner = tournament_sim.simulate_game(game['gameID'])
                self.winners[game['gameID']] = winner
                print(f"Winner: {winner}")
                
                # Process each active player's pick
                for player_id in list(active_players_on_date):
                    if player_id in self.eliminated_players:
                        continue
                        
                    pick = player_picks.get(player_id)
                    if pick is None:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        print(f"Player {player_id} eliminated (no pick available)")
                        continue
                        
                    # Get the teams playing in this game
                    team_a = tournament_sim.get_team_name(game['teamA'])
                    team_b = tournament_sim.get_team_name(game['teamB'])
                    
                    # Only eliminate player if their picked team lost in this game
                    if pick in [team_a, team_b] and pick != winner:
                        self.eliminated_players.add(player_id)
                        self.elimination_dates[player_id] = current_date
                        print(f"Player {player_id} eliminated (picked {pick}, winner was {winner})")
                    elif pick == winner and current_date == self.target_date:
                        print(f"Player {player_id} survived with pick {pick}")
            
            # Check if we have a winner after all games for the day
            active_count = len(self.active_players - self.eliminated_players)
            print(f"\nAfter processing all games for {current_date}:")
            print(f"Active players: {active_count}")
            print(f"Eliminated players: {len(self.eliminated_players)}")
            
            if active_count == 0:
                # All players have been eliminated, determine winner based on who lasted longest
                self.winner = self.determine_winner()
                print(f"Pool complete! Winner: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            elif active_count == 1:
                # One player still active, they win
                self.winner = list(self.active_players - self.eliminated_players)[0]
                print(f"Pool complete! Winner: {self.winner}")
                # Record triumph for the winner
                if self.winner in self.target_date_picks:
                    self.target_date_triumphs.add(self.winner)
                break
            
            current_date = get_next_date(current_date, self.schedule_df)
            if current_date is None:
                # Reached end of schedule, determine winner based on who lasted longest
                self.winner = self.determine_winner()
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

def run_simulations(num_simulations, target_date, target_player, favorites_factor, variance_factor):
    schedule_df, teams_df, picks_df, team_mapping = load_data()
    
    # Initialize counters for statistics
    player_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0})
    team_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0})
    target_player_team_stats = defaultdict(lambda: {'picks': 0, 'triumphs': 0})
    tournament_winners = defaultdict(int)
    
    for sim in range(num_simulations):
        print(f"\nSimulation {sim + 1}/{num_simulations}\n")
        
        # Create new simulators for this simulation
        tournament_sim = TournamentSimulator(schedule_df, teams_df, team_mapping)
        survivor_sim = SurvivorPoolSimulator(schedule_df, teams_df, picks_df, favorites_factor, variance_factor, target_date)
        
        # Run the simulation
        survivor_sim.run_simulation(tournament_sim)
        
        # Make sure we simulate the final game if it hasn't been simulated yet
        final_game = schedule_df[schedule_df['gameID'] == 'FINAL'].iloc[0]
        if 'FINAL' not in tournament_sim.winners:
            winner = tournament_sim.simulate_game('FINAL')
        else:
            winner = tournament_sim.winners['FINAL']
            
        # Record tournament winner
        tournament_winners[winner] += 1
        
        # Record statistics
        for player_id, pick in survivor_sim.target_date_picks.items():
            player_stats[player_id]['picks'] += 1
            team_stats[pick]['picks'] += 1
            
            if player_id in survivor_sim.target_date_triumphs:
                player_stats[player_id]['triumphs'] += 1
                team_stats[pick]['triumphs'] += 1
                
                if player_id == target_player:
                    target_player_team_stats[pick]['triumphs'] += 1
        
        if target_player in survivor_sim.target_date_picks:
            target_player_team_stats[survivor_sim.target_date_picks[target_player]]['picks'] += 1
    
    # Print statistics
    print("\nPlayer Statistics:")
    print("Player ID | Picks on Target Date | Total Triumphs | Triumph Probability")
    print("-" * 75)
    for player_id in sorted(player_stats.keys()):
        stats = player_stats[player_id]
        prob = (stats['triumphs'] / stats['picks'] * 100) if stats['picks'] > 0 else 0
        print(f"{player_id:9d} | {stats['picks']:20d} | {stats['triumphs']:14d} | {prob:8.2f}%")
    print("-" * 75)
    print(f"TOTAL     | {sum(s['picks'] for s in player_stats.values()):20d} | {sum(s['triumphs'] for s in player_stats.values()):14d} | {100:8.2f}%")
    
    print("\nTeam Statistics (All Players):")
    print("Team | Picks on Target Date | Triumphs | Triumph Probability")
    print("-" * 60)
    for team in sorted(team_stats.keys()):
        stats = team_stats[team]
        prob = (stats['triumphs'] / stats['picks'] * 100) if stats['picks'] > 0 else 0
        print(f"{team:20s} | {stats['picks']:20d} | {stats['triumphs']:9d} | {prob:8.2f}%")
    print("-" * 60)
    print(f"TOTAL                | {sum(s['picks'] for s in team_stats.values()):20d} | {sum(s['triumphs'] for s in team_stats.values()):9d} | {100:8.2f}%")
    
    print("\nTeam Statistics (Player {}):".format(target_player))
    print("Team | Picks on Target Date | Triumphs | Triumph Probability")
    print("-" * 60)
    for team in sorted(target_player_team_stats.keys()):
        stats = target_player_team_stats[team]
        prob = (stats['triumphs'] / stats['picks'] * 100) if stats['picks'] > 0 else 0
        print(f"{team:20s} | {stats['picks']:20d} | {stats['triumphs']:9d} | {prob:8.2f}%")
    print("-" * 60)
    print(f"TOTAL                | {sum(s['picks'] for s in target_player_team_stats.values()):20d} | {sum(s['triumphs'] for s in target_player_team_stats.values()):9d} | {100:8.2f}%")
    
    print("\nTournament Winners:")
    print("Team | Wins | Win Percentage")
    print("-" * 40)
    for team in sorted(tournament_winners.keys()):
        wins = tournament_winners[team]
        percentage = (wins / num_simulations) * 100
        print(f"{team:20s} | {wins:4d} | {percentage:8.2f}%")
    print("-" * 40)
    print(f"TOTAL                | {num_simulations:4d} | {100:8.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Simulate tournament and survivor pool')
    parser.add_argument('--simulations', type=int, default=20, help='Number of simulations to run')
    parser.add_argument('--target-date', type=str, required=True, help='Target date to analyze (MM/DD/YYYY)')
    parser.add_argument('--target-player', type=int, required=True, help='Target player ID to analyze')
    parser.add_argument('--favorites-factor', type=int, default=5, help='How much to favor teams with better odds (1-10)')
    parser.add_argument('--variance-factor', type=int, default=5, help='How much random variation in picks (1-10)')
    args = parser.parse_args()

    # Read input files
    schedule_df = pd.read_csv('schedule.csv')
    teams_df = pd.read_csv('teams.csv')
    picks_df = pd.read_csv('sample_picks.csv')
    
    # Set player_id as index for picks_df
    picks_df.set_index('player_id', inplace=True)
    
    # Print picks DataFrame info
    print("\nPicks DataFrame Info:")
    print(f"Columns: {picks_df.columns.tolist()}")
    print(f"Index: {picks_df.index.tolist()[:5]}...")
    print(f"Sample picks for target date {args.target_date}:")
    print(picks_df[args.target_date].head())
    
    # Create team mapping
    team_mapping = create_team_mapping(teams_df, picks_df, schedule_df)
    
    print("\nTeam mapping summary:")
    for team, mapped in sorted(team_mapping.items()):
        print(f"{team} -> {mapped}")
    
    # Run simulations
    run_simulations(args.simulations, args.target_date, args.target_player, args.favorites_factor, args.variance_factor)

if __name__ == "__main__":
    main()