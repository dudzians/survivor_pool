import pandas as pd
from datetime import datetime, timedelta

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
    'Unknown': '',  # Convert Unknown to empty string
    '-': ''  # Convert dashes to empty string
}

def convert_picks():
    # Read the CSV file
    df = pd.read_csv('survivor_pool.csv')
    
    # Print initial stats
    print(f"\nInitial stats:")
    print(f"Total rows: {len(df)}")
    print(f"Unique EntryIDs: {len(df['EntryID'].unique())}")
    print(f"Statuses present: {df['Status'].unique()}")
    
    # Filter out Unknown picks before processing
    df = df[df['Current_Pick'] != 'Unknown']
    
    # Convert abbreviated team names to full names
    df['Current_Pick'] = df['Current_Pick'].map(TEAM_NAMES)
    
    # Remove any empty picks after conversion
    df = df[df['Current_Pick'] != '']
    
    # Group by EntryID and get picks in order of appearance
    player_picks = df.groupby('EntryID')['Current_Pick'].agg(list).reset_index()
    
    # Create the output DataFrame with the required format
    output_df = pd.DataFrame()
    output_df['player_id'] = range(0, 202)  # Exactly 202 players, starting from 0
    
    # Add columns for each day's picks with dates as column headers
    start_date = datetime(2025, 3, 20)  # First day of tournament
    max_picks = max(len(picks) for picks in player_picks['Current_Pick'])
    
    # Initialize all picks as empty strings
    for i in range(max_picks):
        date = start_date + timedelta(days=i)
        date_col = date.strftime('%Y-%m-%d')
        output_df[date_col] = ''
    
    # Fill in the actual picks for all 202 players
    # Each player's picks are assigned sequentially starting from day 1
    for i in range(min(len(player_picks), 202)):
        picks = player_picks.iloc[i]['Current_Pick']
        # Assign picks sequentially to days, starting from day 1
        for j, pick in enumerate(picks):
            date = start_date + timedelta(days=j)  # j starts from 0, so first pick goes to day 1
            date_col = date.strftime('%Y-%m-%d')
            output_df.at[i, date_col] = pick
    
    # Save to CSV
    output_df.to_csv('picks_output.csv', index=False)
    print("\nConversion complete. Check picks_output.csv")
    print("\nFinal stats:")
    print(f"Total players: {len(output_df)}")
    print(f"Total days: {max_picks}")
    
    # Print pick distributions for all days
    for i in range(max_picks):
        date = start_date + timedelta(days=i)
        date_col = date.strftime('%Y-%m-%d')
        print(f"\nPick distribution for {date_col}:")
        print(output_df[date_col].value_counts(dropna=False))

if __name__ == "__main__":
    convert_picks() 