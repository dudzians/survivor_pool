# Tournament Survivor Pool Simulator

This program simulates a tournament survivor pool and determines optimal team picks for each day to maximize the chance of winning the pool.

## Features

- Simulates tournament games based on provided team odds
- Models player behavior using favorites and variance factors
- Runs Monte Carlo simulations to determine optimal picks
- Handles pool rules including one-team-per-player and daily elimination
- Uses seed sum tiebreaker for tied players

## Requirements

- Python 3.8+
- pandas
- numpy

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Input Files

### schedule.csv
Contains the tournament schedule with columns:
- gameID: Unique identifier for each game
- round: Tournament round number (64, 32, 16, 8, 4, 2)
- day: Game date (YYYY-MM-DD format)
- teamA: First team name
- teamB: Second team name

### teams.csv
Contains team information with columns:
- teamName: Team name
- seed: Team's tournament seed
- region: Tournament region
- 64odds, 32odds, 16odds, 8odds, 4odds, 2odds: Win probabilities for each round

## Usage

Run the simulator with:

```bash
python main.py --date YYYY-MM-DD [options]
```

### Command Line Options

- `--date`: Target date to analyze (required, format: YYYY-MM-DD)
- `--players`: Number of players in pool (default: 100)
- `--favorites`: Favorites factor 1-10 (default: 5)
  - Lower values favor underdogs more
  - Higher values favor favorites more
- `--variance`: Variance factor 1-10 (default: 5)
  - Lower values make player picks more similar
  - Higher values make player picks more diverse
- `--simulations`: Number of simulations to run (default: 10000)

### Example

```bash
python main.py --date 2025-03-20 --players 200 --favorites 7 --variance 3
```

## Output

The program outputs a table showing each available team for the target date and their probability of leading to a pool win if picked on that day.

Example output:
```
Optimal picks for 2025-03-20:
Team                 Win Probability
-----------------------------------
Houston             12.5%
Tennessee           11.2%
Auburn              10.8%
...
```

## How It Works

1. For each simulation:
   - Simulates entire tournament based on team odds
   - Models other players' picks based on favorites/variance factors
   - Tracks which picks lead to pool wins
   
2. Aggregates results across all simulations to determine:
   - Which teams are most likely to lead to a pool win
   - Win probability for each possible pick

3. Considers factors like:
   - Team's probability of winning current and future games
   - How likely other players are to pick each team
   - Future game availability and matchups 