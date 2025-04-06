import pandas as pd
import numpy as np
import os

# Dictionary to convert abbreviations to full team names
TEAM_NAMES = {
    'GONZ': 'Gonzaga',
    'ARIZ': 'Arizona',
    'TTU': 'Texas Tech',
    'MD': 'Maryland',
    'SJU': "St. John's",
    'UK': 'Kentucky',
    'PUR': 'Purdue',
    'ISU': 'Iowa State',
    'FLA': 'Florida',
    'WIS': 'Wisconsin',
    'CSU': 'Colorado State',
    'TENN': 'Tennessee',
    'ORE': 'Oregon',
    'MSU': 'Michigan State',
    'CONN': 'Connecticut',
    'SMC': "Saint Mary's",
    'ALA': 'Alabama',
    'ILL': 'Illinois',
    'HOU': 'Houston',
    'AUB': 'Auburn',
    'TXAM': 'Texas A&M',
    'BYU': 'BYU',
    'UCLA': 'UCLA',
    'CLEM': 'Clemson',
    'MIZZ': 'Missouri',
    'UNC': 'North Carolina',
    'KU': 'Kansas',
    'MARQ': 'Marquette',
    'CREI': 'Creighton',
    'BAY': 'Baylor',
    'MISS': 'Mississippi',
    'MCNS': 'McNeese State',
    'UCSD': 'UC San Diego',
    'LOU': 'Louisville',
    'DRKE': 'Drake',
    'MSST': 'Mississippi State',
    'MICH': 'Michigan',
    'MEM': 'Memphis',
    'DUKE': 'Duke',
    'XAV': 'Xavier',
    'LIB': 'Liberty',
    'OKLA': 'Oklahoma',
    'GC': 'Grand Canyon',
    'VAN': 'Vanderbilt',
    'UNM': 'New Mexico',
    'HP': 'High Point',
    'USU': 'Utah State',
    'UGA': 'Georgia',
    'VCU': 'VCU',
    'YALE': 'Yale',
    'ARK': 'Arkansas',
    'Unknown': '',  # Convert Unknown to empty string
    '-': ''  # Convert dashes to empty string
}

def convert_picks():
    # Read the schedule file to get the dates
    schedule_df = pd.read_csv('schedule.csv')
    unique_days = sorted(schedule_df['day'].unique())
    
    # Read the survivor pool picks file
    survivor_df = pd.read_csv('survivor_pool.csv')
    
    # Get sorted list of unique EntryIDs to ensure consistent ordering
    entry_ids = sorted(survivor_df['EntryID'].unique())
    
    # Create output DataFrame with player_id and unique days as columns
    output_df = pd.DataFrame(columns=['player_id'] + unique_days)
    
    # Process each unique EntryID
    for idx, entry_id in enumerate(entry_ids):
        # Get all rows for this player, but only keep the first occurrence of each EntryID
        entry_rows = survivor_df[survivor_df['EntryID'] == entry_id].drop_duplicates(subset=['EntryID', 'Current_Pick', 'Additional_Info'])
        
        # Create new row for this player using index as player_id
        new_row = {'player_id': idx}
        
        # Get all picks for this player
        picks = []
        for _, row in entry_rows.iterrows():
            # Process Current_Pick
            if pd.notna(row['Current_Pick']) and row['Current_Pick'] not in ['', '-']:
                picks.append(TEAM_NAMES.get(row['Current_Pick'], row['Current_Pick']))
            # Process Additional_Info as a separate pick
            if pd.notna(row['Additional_Info']) and row['Additional_Info'] != '' and row['Additional_Info'] != '-':
                picks.append(TEAM_NAMES.get(row['Additional_Info'], row['Additional_Info']))
        
        # Assign picks to days
        for i, day in enumerate(unique_days):
            new_row[day] = picks[i] if i < len(picks) else ''
        
        # Add the row to output DataFrame
        output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save to temporary file first
    temp_file = 'temp_picks.csv'
    output_df.to_csv(temp_file, index=False)
    
    # Try to rename the temp file to the final file
    try:
        if os.path.exists('sample_picks.csv'):
            os.remove('sample_picks.csv')
        os.rename(temp_file, 'sample_picks.csv')
    except Exception as e:
        print(f"Warning: Could not rename temp file to sample_picks.csv. Error: {e}")
        print(f"Data has been saved to {temp_file}")
    
    # Print final stats
    print("\nFinal stats:")
    print(f"Total players: {len(output_df)}")
    print(f"Total days: {len(unique_days)}")
    
    # Print pick distribution for each day
    print("\nPick distribution for each day:\n")
    for day in unique_days:
        print(f"{day}")
        value_counts = output_df[day].value_counts()
        # Don't count empty strings in the value counts
        if '' in value_counts:
            value_counts = value_counts[value_counts.index != '']
        print(value_counts)
        print()

if __name__ == "__main__":
    convert_picks() 