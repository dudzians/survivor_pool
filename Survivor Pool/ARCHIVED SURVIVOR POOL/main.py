import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from data_loader import load_schedule, load_team_odds, get_games_for_date, validate_data, get_teams_playing_on_date
from tournament_simulator import TournamentSimulator
from player_behavior import PoolSimulator
from player_picks import PlayerPicksManager
import argparse
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time
import os
import logging
from team_utils import normalize_team_name
import sys
import random
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Only show the message without timestamp or level
)

def run_simulation_wrapper(args):
    """Wrapper function for multiprocessing that unpacks arguments."""
    return run_single_simulation(**args)

def run_single_simulation(
    schedule: pd.DataFrame,
    team_odds: pd.DataFrame,
    num_players: int,
    favorites_factor: int,
    variance_factor: int,
    target_date: str,
    picks_manager: PlayerPicksManager = None,
    simulation_id: int = 0,
    total_simulations: int = 0,
    track_player: int = 66
) -> Tuple[int, str, str]:
    """Run a single simulation and return the winner ID, target date pick, and tracked player's pick."""
    try:
        # Simulate tournament
        tournament = TournamentSimulator(schedule, team_odds)
        tournament_results = tournament.simulate_tournament()
        
        # Initialize pool
        pool = PoolSimulator(num_players, favorites_factor, variance_factor,
                          schedule, team_odds, picks_manager)
        
        # Track target date pick for winner
        target_date_pick = None
        tracked_player_pick = None
        
        # Get sorted tournament days once
        tournament_days = schedule.sort_values('day')['day'].unique()
        
        # Simulate all days
        surviving_players = list(range(num_players))
        last_surviving_players = surviving_players
        target_date_dt = pd.to_datetime(target_date).strftime('%-m/%-d/%Y')
        
        for current_date in tournament_days:
            day_str = current_date.strftime('%Y-%m-%d')
            day_games = schedule[schedule['day'] == current_date]
            round_num = str(day_games.iloc[0]['round'])
            
            # Get teams playing on this day
            available_teams = tournament.get_teams_playing_on_date(day_str)
            if not available_teams:
                continue
                
            # Simulate picks for this day
            team_picks = pool.simulate_player_picks(day_str, available_teams, round_num)
            
            # Process results
            winners = tournament.get_winners_on_date(day_str)
            surviving_players = pool.process_results(current_date, winners)
            
            # Track the tracked player's pick on target date
            if current_date == target_date_dt:
                if pool.players[track_player].eliminated:
                    tracked_player_pick = "ELIMINATED"
                else:
                    pick = pool.players[track_player].picks.get(day_str)
                    if pick:
                        tracked_player_pick = pick
            
            if surviving_players:
                last_surviving_players = surviving_players
                
        winner_id = pool.determine_pool_winner(last_surviving_players)
        return winner_id, target_date_pick, tracked_player_pick
        
    except Exception as e:
        logging.error(f"Error in simulation {simulation_id}: {str(e)}")
        return None, None, None

def format_date(date_str):
    dt = pd.to_datetime(date_str)
    return f"{dt.month}/{dt.day}/{dt.year}"

def simulate_pool_day(schedule: pd.DataFrame, team_odds: pd.DataFrame, picks: pd.DataFrame,
                   target_date: str, num_simulations: int = 20, favorites_factor: int = 5,
                   variance_factor: int = 5) -> None:
    """Simulate picks and results for each day of the tournament."""
    print(f"\nSimulating {num_simulations} games for {target_date}")
    print(f"Players: {len(picks)} | Favorites: {favorites_factor} | Variance: {variance_factor}\n")
    
    # Convert target_date to match schedule format
    target_date = pd.to_datetime(target_date)
    
    # Track teams playing on target date
    teams_playing = get_teams_playing_on_date(schedule, target_date)
    if not teams_playing:
        raise ValueError(f"No games found for date {target_date}")
    
    # Track player triumphs and picks
    player_triumphs = defaultdict(int)
    player_picks = defaultdict(list)
    team_success_with_triumph = defaultdict(lambda: defaultdict(int))
    player66_team_triumphs = defaultdict(int)  # Track triumphs specifically for Player 66
    
    # Track progress
    start_time = time.time()
    last_update = start_time
    
    # Run simulations
    for sim in range(num_simulations):
        # Initialize pool simulator
        pool = PoolSimulator(len(picks), favorites_factor, variance_factor, schedule, team_odds)
        
        # Simulate tournament results
        tournament = TournamentSimulator(schedule, team_odds)
        tournament.simulate_tournament()
        
        # Get sorted tournament days
        tournament_days = schedule.sort_values('day')['day'].unique()
        
        # Track surviving players
        surviving_players = list(range(len(picks)))
        last_surviving_players = surviving_players
        target_date_picks = {}
        
        # Simulate each day
        for current_date in tournament_days:
            day_str = current_date.strftime('%Y-%m-%d')
            
            # Get teams playing on this day
            available_teams = tournament.get_teams_playing_on_date(day_str)
            if not available_teams:
                continue
            
            # Get winners for this date
            winners = tournament.get_winners_on_date(day_str)
            
            # Process each player's picks
            for player_id in surviving_players.copy():
                if current_date == target_date:
                    pick = picks[player_id].get(target_date.strftime('%Y-%m-%d'))
                    if pick:
                        player_picks[player_id].append(pick)
                        target_date_picks[player_id] = pick
                
                pick = picks[player_id].get(day_str)
                if not pick or pick not in winners:
                    surviving_players.remove(player_id)
            
            if surviving_players:
                last_surviving_players = surviving_players.copy()
        
        # Determine winner and record triumphs
        if last_surviving_players:
            winner = pool.determine_pool_winner(last_surviving_players)
            if winner is not None:
                player_triumphs[winner] += 1
                # Record if the winner's target date pick was successful
                if winner in target_date_picks:
                    team = target_date_picks[winner]
                    team_success_with_triumph[team]['successes'] += 1
                    # If winner is Player 66, record their team triumph
                    if winner == 66:
                        player66_team_triumphs[team] += 1
    
    # Print final progress
    end_time = time.time()
    total_time = end_time - start_time
    speed = num_simulations / total_time
    print(f"\nCompleted {num_simulations} simulations in {total_time:.1f} seconds ({speed:.1f} sims/sec)\n")
    
    # Print player triumph statistics
    print("Player Triumph Statistics:")
    print("Player   Picks   Triumphs   Rate")
    print("--------------------------------")
    for player_id in range(len(picks)):
        triumphs = player_triumphs[player_id]
        picks = len(player_picks[player_id])
        triumph_rate = triumphs / num_simulations * 100
        print(f"{player_id:<8} {picks:<6} {triumphs:<10} {triumph_rate:>6.1f}%")
    
    # Print target player performance
    print("\nTarget Player 66 Performance:")
    print("Team            Picks  Triumphs   Rate")
    print("---------------------------------------")
    if player_picks[66]:
        pick_counts = Counter(player_picks[66])
        for team, count in pick_counts.items():
            triumphs = player66_team_triumphs[team]  # Use Player 66 specific triumphs
            triumph_rate = (triumphs / num_simulations * 100)
            print(f"{team:<15} {count:<6} {triumphs:<10} {triumph_rate:>6.1f}%")
    
    # Calculate optimal picks based on triumph probability
    print("\nOptimal picks for", target_date.strftime('%Y-%m-%d'))
    print("Team                 Picks   Triumphs   Triumph Probability")
    print("--------------------------------------------------------")
    total_picks = 0
    total_triumphs = 0
    for team in sorted(teams_playing):
        # Count picks across all simulations
        picks = sum(1 for player_picks_list in player_picks.values() for pick in player_picks_list if pick == team)
        total_picks += picks
        triumphs = team_success_with_triumph[team]['successes']
        total_triumphs += triumphs
        triumph_probability = (triumphs / num_simulations) * 100
        print(f"{team:<20} {picks:>5}   {triumphs:>8}   {triumph_probability:>6.1f}%")
    print("--------------------------------------------------------")
    print(f"{'TOTAL':<20} {total_picks:>5}   {total_triumphs:>8}   {(total_triumphs/num_simulations):>6.1f}%")

def simulate_tournament(schedule, team_odds, favorites_factor=5, variance_factor=5):
    # Create a copy of the schedule to track results
    tournament_results = {}
    
    # Simulate each game in order
    for _, game in schedule.iterrows():
        game_id = game['gameID']
        team1 = game['teamA']
        team2 = game['teamB']
        
        # If this is a later round game, get the actual teams from previous results
        if team1.startswith('Winner of:'):
            prev_game = team1.split(':')[1].strip()
            team1 = tournament_results[prev_game]
        if team2.startswith('Winner of:'):
            prev_game = team2.split(':')[1].strip()
            team2 = tournament_results[prev_game]
            
        # Get win probabilities from odds
        round_num = str(game['round'])
        odds_col = f"{round_num}odds"
        
        try:
            team1_odds = float(team_odds[team_odds['teamName'] == team1][odds_col].iloc[0])
            team2_odds = float(team_odds[team_odds['teamName'] == team2][odds_col].iloc[0])
        except (IndexError, ValueError):
            # If team not found or invalid odds, use 0.5
            team1_odds = 0.5
            team2_odds = 0.5
        
        # Adjust odds based on favorites factor
        team1_odds = team1_odds ** (1/favorites_factor)
        team2_odds = team2_odds ** (1/favorites_factor)
        
        # Normalize odds
        total_odds = team1_odds + team2_odds
        if total_odds == 0:
            # If both teams have 0 odds, use 0.5 for each
            team1_prob = 0.5
        else:
            team1_prob = team1_odds / total_odds
        
        # Add random variance
        team1_prob = team1_prob + (random.random() - 0.5) * variance_factor * 0.1
        team1_prob = max(0.1, min(0.9, team1_prob))
        
        # Simulate game
        winner = team1 if random.random() < team1_prob else team2
        tournament_results[game_id] = winner
        
    return tournament_results

def main():
    parser = argparse.ArgumentParser(description='Run survivor pool simulation')
    parser.add_argument('--date', type=str, required=True, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--simulations', type=int, default=20, help='Number of simulations to run')
    parser.add_argument('--picks', type=str, help='Path to picks CSV file')
    parser.add_argument('--favorites', type=int, default=5, help='Favorites factor')
    parser.add_argument('--variance', type=int, default=5, help='Variance factor')
    args = parser.parse_args()

    try:
        # Load data
        schedule = load_schedule('schedule.csv')
        team_odds = load_team_odds('teams.csv')
        
        # Validate data
        validate_data(schedule, team_odds)
        
        # Load player picks if provided
        picks = None
        if args.picks:
            logging.info("Loading picks from sample_picks.csv")
            logging.info("Reading CSV file...")
            df = pd.read_csv(args.picks)
            logging.info(f"Loaded {len(df)} picks for {len(df['player_id'].unique())} players")
            picks_manager = PlayerPicksManager(args.picks)
            picks = picks_manager.picks

        simulate_pool_day(
            schedule=schedule,
            team_odds=team_odds,
            picks=picks,
            target_date=args.date,
            num_simulations=args.simulations,
            favorites_factor=args.favorites,
            variance_factor=args.variance
        )

    except Exception as e:
        logging.error(f"Error running simulation: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 