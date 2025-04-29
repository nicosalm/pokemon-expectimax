import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import itertools
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import time

def create_payoff_matrix(team_a, team_b, win_probabilities):
    """Create a payoff matrix for the 3v3 Pokémon battle."""
    n_a = len(team_a)
    n_b = len(team_b)
    payoff_matrix = np.zeros((n_a, n_b))

    for i, pokemon_a in enumerate(team_a):
        for j, pokemon_b in enumerate(team_b):
            matchup = (pokemon_a, pokemon_b)
            # Default to 50/50 if matchup not found
            payoff_matrix[i, j] = win_probabilities.get(matchup, 0.5)

    return payoff_matrix

def compute_nash_equilibrium(payoff_matrix):
    """Compute the Nash equilibrium for a zero-sum game using linear programming."""
    num_rows, num_cols = payoff_matrix.shape

    # For player 1 (row player)
    c = np.zeros(num_rows + 1)
    c[-1] = -1  # Maximize v

    A_ub = np.zeros((num_cols, num_rows + 1))
    for j in range(num_cols):
        A_ub[j, :-1] = -payoff_matrix[:, j]
        A_ub[j, -1] = 1
    b_ub = np.zeros(num_cols)

    A_eq = np.zeros((1, num_rows + 1))
    A_eq[0, :-1] = 1
    b_eq = np.ones(1)

    bounds = [(0, 1) for _ in range(num_rows)] + [(None, None)]

    try:
        # Solve the linear program
        res1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        p1_strategy = res1.x[:-1]
        v = -res1.fun

        # For player 2 (column player)
        c = np.zeros(num_cols + 1)
        c[-1] = 1  # Minimize u

        A_ub = np.zeros((num_rows, num_cols + 1))
        for i in range(num_rows):
            A_ub[i, :-1] = payoff_matrix[i, :]
            A_ub[i, -1] = -1
        b_ub = np.zeros(num_rows)

        A_eq = np.zeros((1, num_cols + 1))
        A_eq[0, :-1] = 1
        b_eq = np.ones(1)

        bounds = [(0, 1) for _ in range(num_cols)] + [(None, None)]

        # Solve the linear program
        res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        p2_strategy = res2.x[:-1]
        u = res2.fun

        if not np.isclose(u, v, atol=1e-5):
            print(f"Warning: Game values do not match. v = {v}, u = {u}")

        return p1_strategy, p2_strategy, v
    except Exception as e:
        print(f"Error computing Nash equilibrium: {e}")
        return np.ones(num_rows) / num_rows, np.ones(num_cols) / num_cols, 0.5

def get_win_probabilities():
    """Return a dictionary of matchup win probabilities."""
    probabilities = {
        # Format: (pokemon_a, pokemon_b): probability of pokemon_a winning
        ('Primarina', 'Primarina'): 0.5,
        ('Primarina', 'Sylveon'): 0.09583,
        ('Primarina', 'Rillaboom'): 0.0252,
        ('Primarina', 'Heatran'): 1.0,
        ('Primarina', 'Urshifu'): 1.0,
        ('Primarina', 'Zapdos'): 0.0,

        ('Sylveon', 'Primarina'): 0.90417,
        ('Sylveon', 'Sylveon'): 0.5,
        ('Sylveon', 'Rillaboom'): 0.387,
        ('Sylveon', 'Heatran'): 0.0,
        ('Sylveon', 'Urshifu'): 1.0,
        ('Sylveon', 'Zapdos'): 0.00625,

        ('Rillaboom', 'Primarina'): 0.9748,
        ('Rillaboom', 'Sylveon'): 0.613,
        ('Rillaboom', 'Rillaboom'): 0.5,
        ('Rillaboom', 'Heatran'): 0.95,
        ('Rillaboom', 'Urshifu'): 0.6954,
        ('Rillaboom', 'Zapdos'): 0.0165,

        ('Heatran', 'Primarina'): 0.0,
        ('Heatran', 'Sylveon'): 1.0,
        ('Heatran', 'Rillaboom'): 0.05,
        ('Heatran', 'Heatran'): 0.5,
        ('Heatran', 'Urshifu'): 0.0,
        ('Heatran', 'Zapdos'): 0.0,

        ('Urshifu', 'Primarina'): 0.0,
        ('Urshifu', 'Sylveon'): 0.0,
        ('Urshifu', 'Rillaboom'): 0.3046,
        ('Urshifu', 'Heatran'): 1.0,
        ('Urshifu', 'Urshifu'): 0.5,
        ('Urshifu', 'Zapdos'): 0.3,

        ('Zapdos', 'Primarina'): 1.0,
        ('Zapdos', 'Sylveon'): 0.99375,
        ('Zapdos', 'Rillaboom'): 0.9835,
        ('Zapdos', 'Heatran'): 1.0,
        ('Zapdos', 'Urshifu'): 0.7,
        ('Zapdos', 'Zapdos'): 0.5
    }
    return probabilities

def analyze_matchup(team_a, team_b, win_probabilities):
    """Analyze a Pokémon matchup and return the game value."""
    # Create payoff matrix
    payoff_matrix = create_payoff_matrix(team_a, team_b, win_probabilities)

    # Compute Nash equilibrium
    _, _, game_value = compute_nash_equilibrium(payoff_matrix)

    return game_value

def generate_all_team_combinations(pokemon_list, team_size=3):
    """Generate all possible team combinations of the given size."""
    return list(itertools.combinations(pokemon_list, team_size))

def format_team(team):
    """Convert a team tuple to a string representation."""
    return ', '.join(team)

def create_performance_table(team_results, top_n=10, bottom_n=10):
    """
    Create a table showing team performance.

    Args:
        team_results: Dictionary mapping team to average game value
        top_n: Number of top teams to display
        bottom_n: Number of bottom teams to display

    Returns:
        DataFrame with team performance data
    """
    # Convert results to DataFrame for easy sorting and display
    df = pd.DataFrame({
        'Team': [format_team(team) for team in team_results.keys()],
        'Avg Win Rate': [game_value for game_value in team_results.values()]
    })

    # Sort by win rate
    df = df.sort_values('Avg Win Rate', ascending=False).reset_index(drop=True)

    # Add rank column
    df.index = df.index + 1
    df = df.rename_axis('Rank').reset_index()

    # Separate top and bottom teams
    top_teams = df.head(top_n).copy()
    bottom_teams = df.tail(bottom_n).copy()

    # Add a separator
    separator = pd.DataFrame({
        'Rank': ['...'],
        'Team': ['...'],
        'Avg Win Rate': [np.nan]
    })

    # Combine into a single table
    if len(df) > (top_n + bottom_n):
        result_table = pd.concat([top_teams, separator, bottom_teams]).reset_index(drop=True)
    else:
        result_table = df

    # Format the win rate as percentage
    result_table['Avg Win Rate'] = result_table['Avg Win Rate'].map(lambda x: f"{x:.1%}" if pd.notnull(x) else x)

    return result_table

def plot_performance_table(df, output_filename, highlight_rows=None):
    """
    Create a color-coded visualization of the performance table.

    Args:
        df: DataFrame with team performance data
        output_filename: Name of the output file
        highlight_rows: List of row indices to highlight
    """
    if highlight_rows is None:
        highlight_rows = []

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc='center', cellLoc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color code the cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_facecolor('#CCCCCC')
            cell.set_text_props(weight='bold')
        elif col == 2 and row - 1 < len(df):  # Win rate column (excluding the separator)
            try:
                win_rate_text = df.iloc[row-1, 2]
                if win_rate_text != '...':
                    win_rate = float(win_rate_text.strip('%')) / 100
                    # Create a color gradient from red (low) to white (neutral) to green (high)
                    if win_rate < 0.5:
                        # Red to white gradient for win rates < 50%
                        intensity = 1 - (win_rate / 0.5)  # 1 at 0%, 0 at 50%
                        cell.set_facecolor((1, 1-intensity*0.5, 1-intensity*0.5))
                    else:
                        # White to green gradient for win rates > 50%
                        intensity = (win_rate - 0.5) / 0.5  # 0 at 50%, 1 at 100%
                        cell.set_facecolor((1-intensity*0.5, 1, 1-intensity*0.5))
            except (ValueError, TypeError):
                pass

    # Highlight specific rows
    for row_idx in highlight_rows:
        for col in range(3):
            if row_idx < len(df):
                table[(row_idx+1, col)].set_facecolor('#FFFF00')  # Yellow highlight

    plt.suptitle('Pokémon 3v3 Team Performance Ranking', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    start_time = time.time()
    print("Starting Pokémon team analysis...")

    # Get win probabilities
    win_probabilities = get_win_probabilities()

    # Get list of all Pokémon
    pokemon_list = sorted(list(set([p[0] for p in win_probabilities.keys()])))
    print(f"Analyzing teams with these Pokémon: {pokemon_list}")

    # Generate all possible team combinations
    teams = generate_all_team_combinations(pokemon_list, team_size=3)
    total_teams = len(teams)
    total_matchups = total_teams * (total_teams - 1) // 2  # Each team vs every other team

    print(f"Generated {total_teams} possible teams.")
    print(f"Will analyze {total_matchups} matchups.")

    # Analyze all matchups
    team_results = {team: 0.0 for team in teams}
    team_matchups = {team: 0 for team in teams}

    total_processed = 0
    last_update = 0

    for i, team_a in enumerate(teams):
        for team_b in teams[i+1:]:  # Only analyze each pair once
            # Compute game value
            game_value = analyze_matchup(team_a, team_b, win_probabilities)

            # Update results for both teams
            team_results[team_a] += game_value
            team_results[team_b] += (1 - game_value)

            team_matchups[team_a] += 1
            team_matchups[team_b] += 1

            total_processed += 1

            # Print progress every 10%
            progress = total_processed / total_matchups
            if progress - last_update >= 0.1:
                elapsed = time.time() - start_time
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed

                print(f"Progress: {progress:.1%} ({total_processed}/{total_matchups})")
                print(f"Time elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
                last_update = progress

    # Compute average win rate for each team
    for team in teams:
        team_results[team] /= team_matchups[team]

    # Create and display the performance table
    performance_table = create_performance_table(team_results, top_n=10, bottom_n=10)
    print("\nTeam Performance Ranking:")
    print(performance_table)

    # Highlight the best and worst team
    highlight_rows = [0, len(performance_table)-1]

    # Save the performance table as an image
    plot_performance_table(performance_table, 'team_performance_ranking.png', highlight_rows)

    # Get the best and worst teams
    best_team = max(team_results, key=team_results.get)
    worst_team = min(team_results, key=team_results.get)

    # Compare head-to-head
    direct_matchup = analyze_matchup(best_team, worst_team, win_probabilities)

    print(f"\nBest Team: {format_team(best_team)}")
    print(f"Average Win Rate: {team_results[best_team]:.1%}")

    print(f"\nWorst Team: {format_team(worst_team)}")
    print(f"Average Win Rate: {team_results[worst_team]:.1%}")

    print(f"\nDirect Matchup (Best vs Worst): Best team wins {direct_matchup:.1%} of the time")

    # Save detailed results to CSV
    all_results_df = pd.DataFrame({
        'Team': [format_team(team) for team in team_results.keys()],
        'Avg Win Rate': [game_value for game_value in team_results.values()]
    }).sort_values('Avg Win Rate', ascending=False)

    all_results_df.to_csv('all_team_results.csv', index=False)
    print("\nDetailed results saved to 'all_team_results.csv'")

    print(f"\nTotal analysis time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
