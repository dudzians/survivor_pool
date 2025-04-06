import pandas as pd
import numpy as np
from datetime import datetime
import random
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Load data
def load_data():
    teams_df = pd.read_csv('teams.csv')
    picks_df = pd.read_csv('sample_picks.csv')
    schedule_df = pd.read_csv('schedule.csv')
    
    print("\nTeams in teams.csv:")
    print(teams_df['teamName'].tolist())
    print("\nUnique teams in schedule.csv:")
    print(set(schedule_df['teamA'].unique()) | set(schedule_df['teamB'].unique()))
    print("\nUnique teams in picks:")
    pick_teams = set()
    for col in picks_df.columns[1:]:
        pick_teams.update(picks_df[col].dropna().unique())
    print(pick_teams)
    
    return teams_df, picks_df, schedule_df

# Team name mapping to handle variations
def create_team_mapping(teams_df, picks_df, schedule_df):
    # Print DataFrame information for debugging
    print("\nTeams DataFrame columns:", teams_df.columns.tolist())
    print("Picks DataFrame columns:", picks_df.columns.tolist())
    print("Schedule DataFrame columns:", schedule_df.columns.tolist())
    
    # Create a mapping of team names to their official names
    team_mapping = {}
    
    # Standard mapping for common variations
    standard_mapping = {
        'Michigan State': 'Michigan St.',
        'Michigan St': 'Michigan St.',
        'Mississippi State': 'Mississippi St.',
        'Mississippi St': 'Mississippi St.',
        'Iowa State': 'Iowa St.',
        'Iowa St': 'Iowa St.',
        'Colorado State': 'Colorado St.',
        'Colorado St': 'Colorado St.',
        'McNeese State': 'McNeese St.',
        'McNeese St': 'McNeese St.',
        'Mount St. Mary\'s': "Mount St. Mary's",
        'Mount St Mary\'s': "Mount St. Mary's",
        'Saint Mary\'s': "Saint Mary's",
        'St. Mary\'s': "Saint Mary's",
        'St Mary\'s': "Saint Mary's",
        'St. John\'s': "St. John's",
        'St John\'s': "St. John's",
        'Alabama State': 'Alabama St.',
        'Alabama St': 'Alabama St.',
        'Norfolk State': 'Norfolk St.',
        'Norfolk St': 'Norfolk St.',
        'Utah State': 'Utah St.',
        'Utah St': 'Utah St.'
    }
    
    # Get all unique team names from all sources
    all_teams = set()
    all_teams.update(teams_df['teamName'].unique())
    
    # Get team names from picks_df (all columns except player_id)
    for col in picks_df.columns[1:]:  # Skip player_id column
        all_teams.update(picks_df[col].dropna().unique())
    
    # Get team names from schedule_df
    all_teams.update(schedule_df['teamA'].unique())
    all_teams.update(schedule_df['teamB'].unique())
    
    # First, map teams that match exactly
    for team in all_teams:
        if pd.isna(team) or str(team).startswith('Winner of:'):
            continue
            
        if team in teams_df['teamName'].values:
            team_mapping[team] = team
        elif team in standard_mapping:
            team_mapping[team] = standard_mapping[team]
    
    # Then, try to map teams that don't match exactly
    for team in all_teams:
        if pd.isna(team) or str(team).startswith('Winner of:'):
            continue
            
        if team not in team_mapping:
            # Try to find a match by removing common suffixes and spaces
            base_name = team.replace(' State', '').replace(' St.', '').replace(' St', '').replace(' ', '').lower()
            for official_team in teams_df['teamName'].values:
                official_base = official_team.replace(' State', '').replace(' St.', '').replace(' St', '').replace(' ', '').lower()
                if base_name == official_base:
                    team_mapping[team] = official_team
                    break
    
    # Print mapping summary
    print("\nTeam Mapping Summary:")
    for original, mapped in sorted(team_mapping.items()):
        if original != mapped:
            print(f"{original:30} -> {mapped}")
    
    # Print warnings for unmapped teams
    unmapped = [team for team in all_teams if team not in team_mapping and not str(team).startswith('Winner of:') and not pd.isna(team)]
    if unmapped:
        print("\nWarning: The following teams could not be mapped:")
        for team in sorted(unmapped):
            print(f"- {team}")
    
    return team_mapping

class TournamentSimulator:
    def __init__(self, schedule_df, teams_df, team_mapping):
        self.schedule_df = schedule_df
        self.teams_df = teams_df
        self.team_mapping = team_mapping
        self.game_results = {}
        
    def simulate_game(self, teamA, teamB):
        if pd.isna(teamA) or pd.isna(teamB):
            return None
            
        if str(teamA).startswith('Winner of:') or str(teamB).startswith('Winner of:'):
            return None
            
        # Map team names to official names
        teamA = self.team_mapping.get(teamA, teamA)
        teamB = self.team_mapping.get(teamB, teamB)
        
        # Get win probabilities from teams_df
        teamA_odds = self.teams_df[self.teams_df['teamName'] == teamA]['64odds'].values[0]
        teamB_odds = self.teams_df[self.teams_df['teamName'] == teamB]['64odds'].values[0]
        
        # Normalize probabilities
        total_odds = teamA_odds + teamB_odds
        teamA_prob = teamA_odds / total_odds
        
        # Simulate game
        return teamA if random.random() < teamA_prob else teamB
        
    def simulate_tournament(self):
        tournament_results = {}
        
        # Simulate each game in schedule
        for _, game in self.schedule_df.iterrows():
            game_id = game['gameID']
            teamA = game['teamA']
            teamB = game['teamB']
            
            # If either team is "Winner of:", look up the winner of that game
            if str(teamA).startswith('Winner of:'):
                prev_game = teamA.split(': ')[1]
                teamA = tournament_results.get(prev_game)
            if str(teamB).startswith('Winner of:'):
                prev_game = teamB.split(': ')[1]
                teamB = tournament_results.get(prev_game)
            
            # Simulate game if we have both teams
            if teamA is not None and teamB is not None:
                winner = self.simulate_game(teamA, teamB)
                tournament_results[game_id] = winner
        
        return tournament_results

class SurvivorPoolSimulator:
    def __init__(self, picks_df, tournament, target_date):
        self.picks_df = picks_df
        self.tournament = tournament
        self.target_date = target_date
        self.active_players = set(picks_df['player_id'])
        self.eliminated_players = set()
        self.player_picks = {}
        
    def get_available_teams(self, player_id, day):
        # Get all teams that have been picked by this player
        used_teams = set()
        for col in self.picks_df.columns[1:]:  # Skip player_id column
            if col <= day:  # Only consider picks up to this day
                team = self.picks_df[self.picks_df['player_id'] == player_id][col].iloc[0]
                if not pd.isna(team):
                    # Map the team name to its official name
                    mapped_team = self.tournament.team_mapping.get(team, team)
                    used_teams.add(mapped_team)
        
        # Get all teams playing on this day
        playing_teams = set()
        day_games = self.tournament.schedule_df[self.tournament.schedule_df['day'] == day]
        for _, game in day_games.iterrows():
            if not pd.isna(game['teamA']) and not str(game['teamA']).startswith('Winner of:'):
                playing_teams.add(game['teamA'])
            if not pd.isna(game['teamB']) and not str(game['teamB']).startswith('Winner of:'):
                playing_teams.add(game['teamB'])
        
        # Return teams that are playing and haven't been used
        return playing_teams - used_teams
        
    def get_player_pick(self, player_id, day):
        # Always check for predetermined pick first
        if day in self.picks_df.columns:
            pick = self.picks_df[self.picks_df['player_id'] == player_id][day].iloc[0]
            if not pd.isna(pick):
                # Map the pick to its official name
                return self.tournament.team_mapping.get(pick, pick)
        
        # If no predetermined pick, check if player is eliminated
        if player_id not in self.active_players:
            return None
        
        # Get available teams for this player
        available_teams = self.get_available_teams(player_id, day)
        if not available_teams:
            self.eliminated_players.add(player_id)
            self.active_players.remove(player_id)
            return None
        
        # Pick team with highest win probability
        best_team = None
        best_odds = -1
        for team in available_teams:
            odds = self.tournament.teams_df[self.tournament.teams_df['teamName'] == team]['64odds'].values[0]
            if odds > best_odds:
                best_odds = odds
                best_team = team
        
        return best_team
        
    def simulate_day(self, day):
        # Get all games for this day
        day_games = self.tournament.schedule_df[self.tournament.schedule_df['day'] == day]
        
        # Track winners and losers
        winners = set()
        losers = set()
        
        # Simulate each game
        for _, game in day_games.iterrows():
            if pd.isna(game['teamA']) or pd.isna(game['teamB']):
                continue
                
            if str(game['teamA']).startswith('Winner of:') or str(game['teamB']).startswith('Winner of:'):
                continue
                
            winner = self.tournament.simulate_game(game['teamA'], game['teamB'])
            if winner == game['teamA']:
                winners.add(game['teamA'])
                losers.add(game['teamB'])
            else:
                winners.add(game['teamB'])
                losers.add(game['teamA'])
        
        # Check each active player's pick
        for player_id in list(self.active_players):
            pick = self.get_player_pick(player_id, day)
            if pick is None:
                self.eliminated_players.add(player_id)
                self.active_players.remove(player_id)
            elif pick in losers:
                self.eliminated_players.add(player_id)
                self.active_players.remove(player_id)
            
            # Store the pick
            if player_id not in self.player_picks:
                self.player_picks[player_id] = {}
            self.player_picks[player_id][day] = pick
        
    def simulate(self):
        # Simulate each day in order
        for day in sorted(self.picks_df.columns[1:]):  # Skip player_id column
            self.simulate_day(day)
            
            # If we've simulated up to target_date, we can stop
            if day >= self.target_date:
                break
                
            # If only one player remains, they are the winner
            if len(self.active_players) == 1:
                return list(self.active_players)[0]
            # If no players remain, pick a random eliminated player as winner
            elif len(self.active_players) == 0:
                return random.choice(list(self.eliminated_players))
        
        # If multiple players are still active, pick a random one as winner
        if len(self.active_players) > 0:
            return random.choice(list(self.active_players))
        else:
            return random.choice(list(self.eliminated_players))

def run_simulation():
    # Load data
    teams_df, picks_df, schedule_df = load_data()
    
    # Create team mapping
    team_mapping = create_team_mapping(teams_df, picks_df, schedule_df)
    
    # Apply team mapping to schedule_df
    schedule_df['teamA'] = schedule_df['teamA'].map(lambda x: team_mapping.get(x, x) if not pd.isna(x) and not str(x).startswith('Winner of:') else x)
    schedule_df['teamB'] = schedule_df['teamB'].map(lambda x: team_mapping.get(x, x) if not pd.isna(x) and not str(x).startswith('Winner of:') else x)
    
    # Apply team mapping to picks_df
    for col in picks_df.columns[1:]:  # Skip player_id column
        picks_df[col] = picks_df[col].map(lambda x: team_mapping.get(x, x) if not pd.isna(x) else x)
    
    # Run simulations
    num_simulations = 100
    target_date = '3/20/2025'
    target_player = 66
    
    # Initialize results storage
    results = {
        'player_stats': defaultdict(lambda: {'picks': 0, 'triumphs': 0}),
        'team_stats': defaultdict(lambda: {'picks': 0, 'triumphs': 0}),
        'target_player_stats': defaultdict(lambda: {'picks': 0, 'triumphs': 0}),
        'player_picks': {},
        'player_triumphs': {}
    }
    
    # Get all player IDs from picks_df
    all_player_ids = sorted(picks_df['player_id'].unique())
    num_players = len(all_player_ids)
    print(f"\nTotal number of players: {num_players}")
    
    for sim_num in range(num_simulations):
        # Create simulator instances
        tournament = TournamentSimulator(schedule_df, teams_df, team_mapping)
        survivor_pool = SurvivorPoolSimulator(picks_df, tournament, target_date)
        
        # Run simulation
        winner = survivor_pool.simulate()
        
        # Track picks for this simulation
        sim_picks = defaultdict(int)
        
        # Update player statistics
        for player_id in all_player_ids:
            # Only track picks on target date
            pick = survivor_pool.get_player_pick(player_id, target_date)
            if pick is not None:
                results['player_stats'][player_id]['picks'] += 1
                sim_picks[pick] += 1
                
                # Update target player statistics
                if player_id == target_player:
                    results['target_player_stats'][pick]['picks'] += 1
                    if player_id == winner:
                        results['target_player_stats'][pick]['triumphs'] += 1
                
                # Update triumphs
                if player_id == winner:
                    results['player_stats'][player_id]['triumphs'] += 1
                    results['team_stats'][pick]['triumphs'] += 1
            
            # Store player picks and triumphs
            if player_id not in results['player_picks']:
                results['player_picks'][player_id] = {}
            results['player_picks'][player_id][sim_num] = pick
            
            # Store player triumphs
            if player_id not in results['player_triumphs']:
                results['player_triumphs'][player_id] = 0
            if player_id == winner:
                results['player_triumphs'][player_id] += 1
        
        # Update team statistics after each simulation
        for team, count in sim_picks.items():
            results['team_stats'][team]['picks'] += count
    
    # Process player statistics
    for player_id in picks_df['player_id'].unique():
        player_triumphs = results['player_triumphs'].get(player_id, 0)
        
        # Each player makes one pick per simulation on the target date
        picks_on_target = num_simulations
        
        results['player_stats'][player_id] = {
            'picks': picks_on_target,
            'triumphs': player_triumphs,
            'triumph_probability': player_triumphs / num_simulations
        }
    
    # Print results
    print("\nResults from {} simulations:".format(num_simulations))
    print("Target date: {}".format(target_date))
    print("Target player: {}".format(target_player))
    
    # Print player statistics
    print("Player Statistics:")
    print("Player ID | Picks on Target Date | Total Triumphs | Triumph Probability")
    print("-" * 75)
    total_picks = 0
    total_triumphs = 0
    for player_id in all_player_ids:
        stats = results['player_stats'][player_id]
        total_picks += stats['picks']
        total_triumphs += stats['triumphs']
        triumph_prob = stats['triumphs'] / num_simulations if stats['picks'] > 0 else 0
        print(f"{player_id:9d} | {stats['picks']:18d} | {stats['triumphs']:14d} | {triumph_prob:.4f}")
    print("-" * 75)
    print(f"TOTAL     | {total_picks:18d} | {total_triumphs:14d} | {total_triumphs/(total_picks if total_picks > 0 else 1):.4f}")
    
    # Print team statistics for all players
    print("\nTeam Statistics (All Players):")
    print("Team | Picks on Target Date | Triumphs | Triumph Probability")
    print("-" * 60)
    total_picks = 0
    total_triumphs = 0
    for team in sorted(results['team_stats'].keys()):
        stats = results['team_stats'][team]
        total_picks += stats['picks']
        total_triumphs += stats['triumphs']
        triumph_prob = stats['triumphs'] / (num_simulations * num_players)  # Calculate probability based on total possible picks
        print(f"{team:30} | {stats['picks']:18d} | {stats['triumphs']:8d} | {triumph_prob:.4f}")
    print("-" * 60)
    print(f"TOTAL                | {total_picks:18d} | {total_triumphs:8d} | {total_triumphs/(num_simulations * num_players):.4f}")
    
    # Print team statistics for target player
    print(f"\nTeam Statistics (Player {target_player}):")
    print("Team | Picks on Target Date | Triumphs | Triumph Probability")
    print("-" * 60)
    total_picks = 0
    total_triumphs = 0
    for team in sorted(results['target_player_stats'].keys()):
        stats = results['target_player_stats'][team]
        total_picks += stats['picks']
        total_triumphs += stats['triumphs']
        triumph_prob = stats['triumphs'] / stats['picks'] if stats['picks'] > 0 else 0
        print(f"{team:30} | {stats['picks']:18d} | {stats['triumphs']:8d} | {triumph_prob:.4f}")
    print("-" * 60)
    print(f"TOTAL                | {total_picks:18d} | {total_triumphs:8d} | {triumph_prob:.4f}")
    
    return results

if __name__ == "__main__":
    # Run simulation with default parameters
    results = run_simulation() 