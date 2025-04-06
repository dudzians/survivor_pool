import json
import pandas as pd
from datetime import datetime, timedelta
import sys
import re

def clean_json_string(json_str):
    # Remove BOM if present
    if json_str.startswith('\ufeff'):
        json_str = json_str[1:]
    
    # Fix the escaped double quotes
    json_str = json_str.replace('""', '"')
    
    # Remove any leading/trailing quotes
    if json_str.startswith('"'):
        json_str = json_str[1:]
    if json_str.endswith('"'):
        json_str = json_str[:-1]
    
    # Fix line endings
    json_str = json_str.replace('\r\n', '\n')
    
    # Remove extra whitespace at the start of lines
    json_str = re.sub(r'^\s+"', '"', json_str, flags=re.MULTILINE)
    
    # Fix any remaining escaped quotes
    json_str = json_str.replace('\\"', '"')
    
    # Remove any extra whitespace
    json_str = json_str.strip()
    
    return json_str

def convert_picks_to_csv():
    try:
        # Read the cleaned JSON file
        with open('cleaned.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a dictionary to store picks by player and date
        picks_by_player = {}
        
        # Process each player's picks
        for player_id, player_data in data.items():
            player_picks = {}
            
            # Go through all picks in all slates
            for slate_id, slate_data in player_data.get('slates', {}).items():
                if 'picks' in slate_data:
                    for pick in slate_data['picks']:
                        game_id = pick.get('gameId', '')
                        team_name = pick.get('teamName', '')
                        game_status = pick.get('gameStatus', '')
                        
                        # Store the pick
                        if game_id and team_name:
                            player_picks[game_id] = {
                                'team': team_name,
                                'status': game_status
                            }
            
            if player_picks:
                picks_by_player[player_id] = player_picks
        
        # Convert to DataFrame
        rows = []
        for player_id, picks in picks_by_player.items():
            row = {'player_id': player_id}
            for game_id, pick_info in picks.items():
                row[game_id] = pick_info['team']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        df.to_csv('converted_picks.csv', index=False)
        print("Conversion complete. Check converted_picks.csv")
        
        # Print sample of the data
        print("\nFirst few rows of converted data:")
        print(df.head())
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        with open('cleaned.json', 'r', encoding='utf-8') as f:
            content = f.read()
            print("First 500 characters of content:")
            print(content[:500])
    except Exception as e:
        print(f"Error: {e}")

def extract_picks():
    picks = []
    current_player = None
    player_picks = {}  # Dictionary to store all picks for each player
    
    with open('splashsports.csv', 'r', encoding='utf-8-sig') as f:
        for line in f:
            # Look for player ID
            player_match = re.search(r'""([0-9a-f-]{36})"":', line)
            if player_match:
                current_player = player_match.group(1)
                if current_player not in player_picks:
                    player_picks[current_player] = []
                continue
            
            # Look for picks
            if '""teamName""' in line and current_player:
                team_match = re.search(r'""teamName"":\s*""([^"]+)""', line)
                if team_match:
                    team_name = team_match.group(1)
                    if team_name not in player_picks[current_player]:
                        player_picks[current_player].append(team_name)
    
    if not player_picks:
        print("No picks found in the file")
        return
    
    # Find the maximum number of picks for any player
    max_picks = max(len(picks) for picks in player_picks.values())
    print(f"\nMaximum picks per player: {max_picks}")
    
    # Create the final format
    rows = []
    for player_id, picks in player_picks.items():
        row = {'player_id': player_id}
        # Add picks for each day, padding with empty strings if needed
        for i in range(max_picks):
            row[f'day_{i+1}'] = picks[i] if i < len(picks) else ''
        rows.append(row)
    
    # Convert to DataFrame
    final_df = pd.DataFrame(rows)
    
    try:
        # Save to CSV
        output_file = 'picks_output.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nConversion complete. Saved to {output_file}")
        print("\nFirst few rows of data:")
        print(final_df.head())
        print(f"\nTotal players: {len(final_df)}")
        print(f"Total days: {max_picks}")
        
        # Print pick counts per player
        print("\nPicks per player:")
        for player_id, picks in player_picks.items():
            if len(picks) > 0:  # Only show players with picks
                print(f"Player {player_id}: {len(picks)} picks - {', '.join(picks)}")
            
    except Exception as e:
        print(f"Error saving file: {e}")
        print("\nDataFrame contents:")
        print(final_df)

if __name__ == "__main__":
    extract_picks() 