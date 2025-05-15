import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import itertools
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import time
from src.pokemon_data import get_win_probabilities

def create_payoff_matrix(team_a, team_b, win_probabilities):
    n_a = len(team_a)
    n_b = len(team_b)
    payoff_matrix = np.zeros((n_a, n_b))

    for i, pokemon_a in enumerate(team_a):
        for j, pokemon_b in enumerate(team_b):
            matchup = (pokemon_a, pokemon_b)
            payoff_matrix[i, j] = win_probabilities.get(matchup, 0.5)

    return payoff_matrix

def compute_nash_equilibrium(payoff_matrix):
    num_rows, num_cols = payoff_matrix.shape

    c = np.zeros(num_rows + 1)
    c[-1] = -1

    a_ub = np.zeros((num_cols, num_rows + 1))
    for j in range(num_cols):
        a_ub[j, :-1] = -payoff_matrix[:, j]
        a_ub[j, -1] = 1
    b_ub = np.zeros(num_cols)

    a_eq = np.zeros((1, num_rows + 1))
    a_eq[0, :-1] = 1
    b_eq = np.ones(1)

    bounds = [(0, 1) for _ in range(num_rows)] + [(None, None)]

    try:
        res1 = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method='highs')

        p1_strategy = res1.x[:-1]
        v = -res1.fun

        c = np.zeros(num_cols + 1)
        c[-1] = 1

        a_ub = np.zeros((num_rows, num_cols + 1))
        for i in range(num_rows):
            a_ub[i, :-1] = payoff_matrix[i, :]
            a_ub[i, -1] = -1
        b_ub = np.zeros(num_rows)

        a_eq = np.zeros((1, num_cols + 1))
        a_eq[0, :-1] = 1
        b_eq = np.ones(1)

        bounds = [(0, 1) for _ in range(num_cols)] + [(None, None)]

        res2 = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method='highs')

        p2_strategy = res2.x[:-1]
        u = res2.fun

        if not np.isclose(u, v, atol=1e-5):
            print(f"warning: game values do not match. v = {v}, u = {u}")

        return p1_strategy, p2_strategy, v
    except Exception as e:
        print(f"error computing nash equilibrium: {e}")
        return np.ones(num_rows) / num_rows, np.ones(num_cols) / num_cols, 0.5

def analyze_matchup(team_a, team_b, win_probabilities):
    payoff_matrix = create_payoff_matrix(team_a, team_b, win_probabilities)
    _, _, game_value = compute_nash_equilibrium(payoff_matrix)
    return game_value

def generate_all_team_combinations(pokemon_list, team_size=3):
    return list(itertools.combinations(pokemon_list, team_size))

def format_team(team):
    return ', '.join(team)

def create_performance_table(team_results, top_n=10, bottom_n=10):
    df = pd.DataFrame({
        'team': [format_team(team) for team in team_results.keys()],
        'avg_win_rate': [game_value for game_value in team_results.values()]
    })

    df = df.sort_values('avg_win_rate', ascending=False).reset_index(drop=True)

    df.index = df.index + 1
    df = df.rename_axis('rank').reset_index()

    top_teams = df.head(top_n).copy()
    bottom_teams = df.tail(bottom_n).copy()

    separator = pd.DataFrame({
        'rank': ['...'],
        'team': ['...'],
        'avg_win_rate': [np.nan]
    })

    if len(df) > (top_n + bottom_n):
        result_table = pd.concat([top_teams, separator, bottom_teams]).reset_index(drop=True)
    else:
        result_table = df

    result_table['avg_win_rate'] = result_table['avg_win_rate'].map(lambda x: f"{x:.1%}" if pd.notnull(x) else x)

    return result_table

def plot_performance_table(df, output_filename, highlight_rows=None):
    if highlight_rows is None:
        highlight_rows = []

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#CCCCCC')
            cell.set_text_props(weight='bold')
        elif col == 2 and row - 1 < len(df):
            try:
                win_rate_text = df.iloc[row-1, 2]
                if win_rate_text != '...':
                    win_rate = float(win_rate_text.strip('%')) / 100
                    if win_rate < 0.5:
                        intensity = 1 - (win_rate / 0.5)
                        cell.set_facecolor((1, 1-intensity*0.5, 1-intensity*0.5))
                    else:
                        intensity = (win_rate - 0.5) / 0.5
                        cell.set_facecolor((1-intensity*0.5, 1, 1-intensity*0.5))
            except (ValueError, TypeError):
                pass

    for row_idx in highlight_rows:
        for col in range(3):
            if row_idx < len(df):
                table[(row_idx+1, col)].set_facecolor('#FFFF00')

    plt.suptitle('pokémon 3v3 team performance ranking', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    start_time = time.time()
    print("starting pokémon team analysis...")

    win_probabilities = get_win_probabilities()

    pokemon_list = sorted(list(set([p[0] for p in win_probabilities.keys()])))
    print(f"analyzing teams with these pokémon: {pokemon_list}")

    teams = generate_all_team_combinations(pokemon_list, team_size=3)
    total_teams = len(teams)
    total_matchups = total_teams * (total_teams - 1) // 2

    print(f"generated {total_teams} possible teams.")
    print(f"will analyze {total_matchups} matchups.")

    team_results = {team: 0.0 for team in teams}
    team_matchups = {team: 0 for team in teams}

    total_processed = 0
    last_update = 0

    for i, team_a in enumerate(teams):
        for team_b in teams[i+1:]:
            game_value = analyze_matchup(team_a, team_b, win_probabilities)

            team_results[team_a] += game_value
            team_results[team_b] += (1 - game_value)

            team_matchups[team_a] += 1
            team_matchups[team_b] += 1

            total_processed += 1

            progress = total_processed / total_matchups
            if progress - last_update >= 0.1:
                elapsed = time.time() - start_time
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed

                print(f"progress: {progress:.1%} ({total_processed}/{total_matchups})")
                print(f"time elapsed: {elapsed:.1f}s, estimated remaining: {remaining:.1f}s")
                last_update = progress

    for team in teams:
        team_results[team] /= team_matchups[team]

    performance_table = create_performance_table(team_results, top_n=17, bottom_n=18)
    print("\nteam performance ranking:")
    print(performance_table)

    highlight_rows = [0, len(performance_table)-1]

    plot_performance_table(performance_table, 'out/team_performance_ranking.png', highlight_rows)

    best_team = max(team_results, key=team_results.get)
    worst_team = min(team_results, key=team_results.get)

    direct_matchup = analyze_matchup(best_team, worst_team, win_probabilities)

    print(f"\nbest team: {format_team(best_team)}")
    print(f"average win rate: {team_results[best_team]:.1%}")

    print(f"\nworst team: {format_team(worst_team)}")
    print(f"average win rate: {team_results[worst_team]:.1%}")

    print(f"\ndirect matchup (best vs worst): best team wins {direct_matchup:.1%} of the time")

    all_results_df = pd.DataFrame({
        'team': [format_team(team) for team in team_results.keys()],
        'avg_win_rate': [game_value for game_value in team_results.values()]
    }).sort_values('avg_win_rate', ascending=False)

    all_results_df.to_csv('out/all_team_results.csv', index=False)
    print("\ndetailed results saved to 'out/all_team_results.csv'")

    print(f"\ntotal analysis time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
