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
import traceback

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
        target_date_dt = format_date_mdy(pd.to_datetime(target_date))
        
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
            if format_date_mdy(current_date) == target_date_dt:
                if pool.players[track_player].eliminated:
                    tracked_player_pick = "ELIMINATED"
                else:
                    pick = pool.players[track_player].picks.get(day_str)
                    if pick:
                        tracked_player_pick = pick
            
            if surviving_players:
                last_surviving_players = surviving_players
                
        winner_id = pool.determine_pool_winner(last_surviving_players)
        
        # Get the winner's pick on target date if they won
        if winner_id is not None:
            target_date_pick = pool.players[winner_id].picks.get(target_date_dt)
            if not target_date_pick:
                target_date_pick = pool.players[winner_id].picks.get(target_date)
        
        return winner_id, target_date_pick, tracked_player_pick
        
    except Exception as e:
        logging.error(f"Error in simulation {simulation_id}: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        logging.error(traceback.format_exc())
        return None, None, None

def format_date(date_str):
    dt = pd.to_datetime(date_str)
    return f"{dt.month}/{dt.day}/{dt.year}"

def format_date_mdy(date_obj):
    """Format date as M/D/YYYY, removing leading zeros."""
    month = str(int(date_obj.strftime('%m')))  # Remove leading zeros
    day = str(int(date_obj.strftime('%d')))    # Remove leading zeros
    year = date_obj.strftime('%Y')
    return f"{month}/{day}/{year}"

def simulate_pool_day(pool, date, winners, target_date, optimal_picks, player_picks, target_player_id):
    """Simulate one day of the survivor pool"""
    # Get all players who need to make picks on this date
    players_to_pick = [p for p in pool.players.values() if p.is_alive and date not in p.picks]
    
    # For each player who needs to make a pick
    for player in players_to_pick:
        # Get available teams (teams that won and haven't been picked by this player)
        available_teams = [team for team in winners if team not in player.picks.values()]
        
        if not available_teams:
            # Player has no teams left to pick from
            player.is_alive = False
            continue
            
        # Make a pick based on favorites and variance factors
        pick = make_pick(available_teams, pool.favorites_factor, pool.variance_factor)
        player.picks[date] = pick
        
        # Track the pick in player_picks
        if player.id == target_player_id:
            if pick not in player_picks:
                player_picks[pick] = {'picks': 0, 'triumphs': 0}
            player_picks[pick]['picks'] += 1
        
        # Check if the pick won
        if pick not in winners:
            player.is_alive = False
        elif date == target_date:
            # Record this pick in optimal_picks if it was made on the target date
            if pick not in optimal_picks:
                optimal_picks[pick] = {'picks': 0, 'triumphs': 0}
            optimal_picks[pick]['picks'] += 1

def simulate_pool(target_date: str, num_simulations: int, picks_file: str, favorites_factor: int = 5, variance_factor: int = 5):
    """Simulate the pool for a specific day."""
    # Load schedule and teams
    schedule = load_schedule('schedule.csv')
    teams = load_team_odds('teams.csv')
    
    # Load picks from file
    picks_manager = PlayerPicksManager(picks_file)
    num_players = picks_manager.num_players
    
    # Initialize tracking dictionaries
    optimal_picks = {}  # {team: {'picks': count, 'triumphs': count}}
    player_picks = {}  # {team: {'picks': count, 'triumphs': count}}
    all_player_stats = {}  # {player_id: {'picks': count, 'triumphs': count}}
    target_player_id = 66  # Hardcoded for now
    
    # Run simulations
    for sim in range(num_simulations):
        winner_id, winner_pick, target_player_pick = run_single_simulation(
            schedule=schedule,
            team_odds=teams,
            num_players=num_players,
            favorites_factor=favorites_factor,
            variance_factor=variance_factor,
            target_date=target_date,
            picks_manager=picks_manager,
            simulation_id=sim,
            total_simulations=num_simulations,
            track_player=target_player_id
        )
        
        # Record results
        if winner_id is not None:
            # Track all player stats
            for player_id in range(num_players):
                if player_id not in all_player_stats:
                    all_player_stats[player_id] = {'picks': 0, 'triumphs': 0}
                all_player_stats[player_id]['picks'] += 1
                if player_id == winner_id:
                    all_player_stats[player_id]['triumphs'] += 1
            
            # Track optimal picks
            if winner_pick:
                if winner_pick not in optimal_picks:
                    optimal_picks[winner_pick] = {'picks': 0, 'triumphs': 0}
                optimal_picks[winner_pick]['picks'] += 1
                # Only increment triumphs if this pick was made by the winner on the target date
                if picks_manager.get_pick(winner_id, target_date) == winner_pick:
                    optimal_picks[winner_pick]['triumphs'] += 1
        
        # Track target player's picks
        if target_player_pick and target_player_pick != "ELIMINATED":
            if target_player_pick not in player_picks:
                player_picks[target_player_pick] = {'picks': 0, 'triumphs': 0}
            player_picks[target_player_pick]['picks'] += 1
            if winner_id == target_player_id:
                player_picks[target_player_pick]['triumphs'] += 1
    
    # Print results
    print("\nAll Players Performance")
    print("Player ID    Picks   Triumphs   Triumph Probability")
    print("-" * 60)
    total_picks = sum(stats['picks'] for stats in all_player_stats.values())
    total_triumphs = sum(stats['triumphs'] for stats in all_player_stats.values())
    for player_id, stats in sorted(all_player_stats.items(), key=lambda x: x[1]['picks'], reverse=True):
        if stats['picks'] > 0:
            triumph_prob = (stats['triumphs'] / stats['picks']) * 100
            print(f"{player_id:>9} {stats['picks']:>6} {stats['triumphs']:>9} {triumph_prob:>6.1f}%")
    print("-" * 60)
    print(f"TOTAL     {total_picks:>6} {total_triumphs:>9} {(total_triumphs/total_picks*100 if total_picks > 0 else 0):>6.1f}%")
    
    print("\nOptimal picks for", target_date)
    print("Team                 Picks   Triumphs   Triumph Probability")
    print("-" * 60)
    total_picks = sum(stats['picks'] for stats in optimal_picks.values())
    total_triumphs = sum(stats['triumphs'] for stats in optimal_picks.values())
    for team, stats in sorted(optimal_picks.items(), key=lambda x: x[1]['picks'], reverse=True):
        if stats['picks'] > 0:
            triumph_prob = (stats['triumphs'] / stats['picks']) * 100
            print(f"{team:<20} {stats['picks']:>6} {stats['triumphs']:>9} {triumph_prob:>6.1f}%")
    print("-" * 60)
    print(f"TOTAL     {total_picks:>6} {total_triumphs:>9} {(total_triumphs/total_picks*100 if total_picks > 0 else 0):>6.1f}%")
    
    print("\nTarget player performance")
    print("Team            Picks  Triumphs   Rate")
    print("-" * 40)
    total_picks = sum(stats['picks'] for stats in player_picks.values())
    total_triumphs = sum(stats['triumphs'] for stats in player_picks.values())
    for team, stats in sorted(player_picks.items(), key=lambda x: x[1]['picks'], reverse=True):
        if stats['picks'] > 0:
            triumph_rate = (stats['triumphs'] / stats['picks']) * 100
            print(f"{team:<15} {stats['picks']:>6} {stats['triumphs']:>9} {triumph_rate:>6.1f}%")
    print("-" * 40)
    print(f"TOTAL     {total_picks:>6} {total_triumphs:>9} {(total_triumphs/total_picks*100 if total_picks > 0 else 0):>6.1f}%")

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
        picks_manager = None
        if args.picks:
            logging.info("Loading picks from sample_picks.csv")
            logging.info("Reading CSV file...")
            picks_manager = PlayerPicksManager(args.picks)
            logging.info(f"Loaded picks for {len(picks_manager.picks)} players")

        simulate_pool(
            target_date=args.date,
            num_simulations=args.simulations,
            picks_file=args.picks,
            favorites_factor=args.favorites,
            variance_factor=args.variance
        )

    except Exception as e:
        logging.error(f"Error running simulation: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 