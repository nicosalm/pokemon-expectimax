import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from src.pokemon_data import get_win_probabilities

class arrow_3d(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

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

def calculate_expected_win_rates(payoff_matrix, p1_strategy, p2_strategy):
    num_rows, num_cols = payoff_matrix.shape

    team_a_win_rates = []
    for i in range(num_rows):
        if p1_strategy[i] < 1e-5:
            team_a_win_rates.append(None)
            continue

        win_rate = 0
        for j in range(num_cols):
            win_rate += payoff_matrix[i, j] * p2_strategy[j]
        team_a_win_rates.append(win_rate)

    team_b_win_rates = []
    for j in range(num_cols):
        if p2_strategy[j] < 1e-5:
            team_b_win_rates.append(None)
            continue

        win_rate = 0
        for i in range(num_rows):
            win_rate += (1 - payoff_matrix[i, j]) * p1_strategy[i]
        team_b_win_rates.append(win_rate)

    return team_a_win_rates, team_b_win_rates

def plot_heatmap(team_a, team_b, payoff_matrix, output_filename):
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('BuWtRd', colors, N=100)

    im = ax.imshow(payoff_matrix, cmap=cmap, vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Win Probability for Team A')

    ax.set_xticks(np.arange(len(team_b)))
    ax.set_yticks(np.arange(len(team_a)))
    ax.set_xticklabels(team_b)
    ax.set_yticklabels(team_a)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(team_a)):
        for j in range(len(team_b)):
            text_color = 'white' if payoff_matrix[i, j] < 0.3 or payoff_matrix[i, j] > 0.7 else 'black'
            ax.text(j, i, f"{payoff_matrix[i, j]:.3f}", ha="center", va="center", color=text_color)

    ax.set_title(f"Battle Matchup Win Probabilities\nTeam A (rows) vs Team B (columns)")

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_simplex(team_a, team_b, p1_strategy, p2_strategy, game_value, output_filename):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0.5, np.sqrt(3)/2, 0])

    for ax in [ax1, ax2]:
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-', linewidth=2)
        ax.plot([v2[0], v3[0]], [v2[1], v3[1]], [v2[2], v3[2]], 'k-', linewidth=2)
        ax.plot([v3[0], v1[0]], [v3[1], v1[1]], [v3[2], v1[2]], 'k-', linewidth=2)

    nash_pos_a = p1_strategy[0] * v1 + p1_strategy[1] * v2 + p1_strategy[2] * v3
    nash_pos_b = p2_strategy[0] * v1 + p2_strategy[1] * v2 + p2_strategy[2] * v3

    ax1.scatter([nash_pos_a[0]], [nash_pos_a[1]], [nash_pos_a[2]], color='red', s=100)
    ax2.scatter([nash_pos_b[0]], [nash_pos_b[1]], [nash_pos_b[2]], color='red', s=100)

    ax1.text(v1[0]-0.1, v1[1]-0.1, v1[2], team_a[0], fontsize=12)
    ax1.text(v2[0]+0.1, v2[1]-0.1, v2[2], team_a[1], fontsize=12)
    ax1.text(v3[0], v3[1]+0.1, v3[2], team_a[2], fontsize=12)

    ax2.text(v1[0]-0.1, v1[1]-0.1, v1[2], team_b[0], fontsize=12)
    ax2.text(v2[0]+0.1, v2[1]-0.1, v2[2], team_b[1], fontsize=12)
    ax2.text(v3[0], v3[1]+0.1, v3[2], team_b[2], fontsize=12)

    ax1.legend([f"Nash Equilibrium\n({p1_strategy[0]:.2f}, {p1_strategy[1]:.2f}, {p1_strategy[2]:.2f})"],
              loc='lower center', bbox_to_anchor=(0.5, -0.15))
    ax2.legend([f"Nash Equilibrium\n({p2_strategy[0]:.2f}, {p2_strategy[1]:.2f}, {p2_strategy[2]:.2f})"],
              loc='lower center', bbox_to_anchor=(0.5, -0.15))

    ax1.set_title(f"Team A Strategy\nOptimal Mixed Strategy", fontsize=14)
    ax2.set_title(f"Team B Strategy\nOptimal Mixed Strategy", fontsize=14)

    ax1.view_init(elev=30, azim=45)
    ax2.view_init(elev=30, azim=45)

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 0.1)

    fig.suptitle(f"3v3 Pokémon Battle Nash Equilibrium\nGame Value: {game_value:.3f}", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_strategies_pie(team_a, team_b, p1_strategy, p2_strategy, game_value, output_filename, team_a_win_rates=None, team_b_win_rates=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    a_labels = []
    b_labels = []

    for i, pokemon in enumerate(team_a):
        if team_a_win_rates and team_a_win_rates[i] is not None:
            a_labels.append(f"{pokemon}\n({team_a_win_rates[i]:.1%} win rate)")
        else:
            a_labels.append(pokemon)

    for i, pokemon in enumerate(team_b):
        if team_b_win_rates and team_b_win_rates[i] is not None:
            b_labels.append(f"{pokemon}\n({team_b_win_rates[i]:.1%} win rate)")
        else:
            b_labels.append(pokemon)

    ax1.pie(p1_strategy, labels=a_labels, autopct='%1.1f%%',
           startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
    ax1.set_title(f"Team A Strategy")

    ax2.pie(p2_strategy, labels=b_labels, autopct='%1.1f%%',
           startangle=90, colors=['#FFCC99', '#C2C2F0', '#FFFF99'])
    ax2.set_title(f"Team B Strategy")

    plt.suptitle(f"Nash Equilibrium Strategies (Game Value: {game_value:.3f})", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def analyze_matchup(team_a, team_b, win_probabilities, output_prefix):
    print(f"\nanalyzing {team_a} vs {team_b}")

    payoff_matrix = create_payoff_matrix(team_a, team_b, win_probabilities)
    print(f"payoff matrix:\n{payoff_matrix}")

    p1_strategy, p2_strategy, game_value = compute_nash_equilibrium(payoff_matrix)

    team_a_win_rates, team_b_win_rates = calculate_expected_win_rates(payoff_matrix, p1_strategy, p2_strategy)

    print(f"team a strategy: {p1_strategy}")
    print(f"team b strategy: {p2_strategy}")
    print(f"game value: {game_value:.3f}")

    print("expected win rates for team a pokémon:")
    for i, pokemon in enumerate(team_a):
        if team_a_win_rates[i] is not None:
            print(f"  {pokemon}: {team_a_win_rates[i]:.1%}")
        else:
            print(f"  {pokemon}: not used")

    print("expected win rates for team b pokémon:")
    for i, pokemon in enumerate(team_b):
        if team_b_win_rates[i] is not None:
            print(f"  {pokemon}: {team_b_win_rates[i]:.1%}")
        else:
            print(f"  {pokemon}: not used")

    plot_heatmap(team_a, team_b, payoff_matrix, f"out/{output_prefix}_heatmap.png")
    plot_simplex(team_a, team_b, p1_strategy, p2_strategy, game_value, f"out/{output_prefix}_simplex.png")
    plot_strategies_pie(team_a, team_b, p1_strategy, p2_strategy, game_value, f"out/{output_prefix}_strategies.png", team_a_win_rates, team_b_win_rates)

    return {
        'team_a': team_a,
        'team_b': team_b,
        'payoff_matrix': payoff_matrix,
        'p1_strategy': p1_strategy,
        'p2_strategy': p2_strategy,
        'game_value': game_value,
        'team_a_win_rates': team_a_win_rates,
        'team_b_win_rates': team_b_win_rates
    }

def is_mixed_strategy(strategy, threshold=0.01):
    non_zero = [prob for prob in strategy if prob > threshold]
    return len(non_zero) > 1

def is_interesting_matchup(p1_strategy, p2_strategy, game_value, threshold=0.01):
    is_mixed = is_mixed_strategy(p1_strategy, threshold) or is_mixed_strategy(p2_strategy, threshold)
    is_balanced = 0.2 <= game_value <= 0.8
    return is_mixed or is_balanced

def create_summary_visualization(matchups, output_filename):
    num_matchups = len(matchups)
    fig, axes = plt.subplots(num_matchups, 3, figsize=(18, 6*num_matchups))

    for i, matchup in enumerate(matchups):
        team_a = matchup['team_a']
        team_b = matchup['team_b']
        p1_strategy = matchup['p1_strategy']
        p2_strategy = matchup['p2_strategy']
        game_value = matchup['game_value']
        team_a_win_rates = matchup.get('team_a_win_rates', None)
        team_b_win_rates = matchup.get('team_b_win_rates', None)

        axes[i, 0].axis('off')
        is_mixed = is_mixed_strategy(p1_strategy) or is_mixed_strategy(p2_strategy)
        strategy_type = "mixed strategy" if is_mixed else "pure strategy"
        summary_text = (
            f"matchup {i+1}: {team_a} vs {team_b}\n\n"
            f"game value: {game_value:.3f}\n\n"
            f"equilibrium type: {strategy_type}\n\n"
            f"team a strategy:\n"
        )
        for j, (pokemon, prob) in enumerate(zip(team_a, p1_strategy)):
            summary_text += f"  {pokemon}: {prob:.1%}"
            if team_a_win_rates and team_a_win_rates[j] is not None:
                summary_text += f" (win rate: {team_a_win_rates[j]:.1%})"
            summary_text += "\n"

        summary_text += f"\nteam b strategy:\n"
        for j, (pokemon, prob) in enumerate(zip(team_b, p2_strategy)):
            summary_text += f"  {pokemon}: {prob:.1%}"
            if team_b_win_rates and team_b_win_rates[j] is not None:
                summary_text += f" (win rate: {team_b_win_rates[j]:.1%})"
            summary_text += "\n"

        axes[i, 0].text(0, 0.5, summary_text, va='center', fontsize=12)

        a_labels = []
        b_labels = []

        for j, pokemon in enumerate(team_a):
            if team_a_win_rates and team_a_win_rates[j] is not None and p1_strategy[j] > 0.01:
                a_labels.append(f"{pokemon}\n({team_a_win_rates[j]:.1%} win rate)")
            else:
                a_labels.append(pokemon)

        for j, pokemon in enumerate(team_b):
            if team_b_win_rates and team_b_win_rates[j] is not None and p2_strategy[j] > 0.01:
                b_labels.append(f"{pokemon}\n({team_b_win_rates[j]:.1%} win rate)")
            else:
                b_labels.append(pokemon)

        axes[i, 1].pie(p1_strategy, labels=a_labels, autopct='%1.1f%%',
                      startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
        axes[i, 1].set_title(f"team a strategy")

        axes[i, 2].pie(p2_strategy, labels=b_labels, autopct='%1.1f%%',
                      startangle=90, colors=['#FFCC99', '#C2C2F0', '#FFFF99'])
        axes[i, 2].set_title(f"team b strategy")

    plt.suptitle("nash equilibrium strategies for 3v3 pokémon battles", fontsize=20)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_expected_win_rate_table(results, output_filename):
    num_matchups = len(results)

    all_pokemon = set()
    for result in results:
        all_pokemon.update(result['team_a'])
        all_pokemon.update(result['team_b'])

    all_pokemon = sorted(list(all_pokemon))
    num_pokemon = len(all_pokemon)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    header = ["matchup", "team", "game value"] + all_pokemon
    table_data.append(header)

    for i, result in enumerate(results):
        row_a = [f"matchup {i+1}", "team a", f"{result['game_value']:.3f}"]
        for pokemon in all_pokemon:
            try:
                idx = result['team_a'].index(pokemon)
                win_rate = result['team_a_win_rates'][idx]
                if win_rate is None:
                    row_a.append("not used")
                else:
                    row_a.append(f"{win_rate:.1%}")
            except ValueError:
                row_a.append("n/a")

        row_b = [f"matchup {i+1}", "team b", f"{1-result['game_value']:.3f}"]
        for pokemon in all_pokemon:
            try:
                idx = result['team_b'].index(pokemon)
                win_rate = result['team_b_win_rates'][idx]
                if win_rate is None:
                    row_b.append("not used")
                else:
                    row_b.append(f"{win_rate:.1%}")
            except ValueError:
                row_b.append("n/a")

        table_data.append(row_a)
        table_data.append(row_b)

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', cellLoc='center', colColours=['#DDDDDD']*len(header))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        elif col >= 3:
            try:
                win_rate_text = cell.get_text().get_text()
                if win_rate_text not in ["n/a", "not used"]:
                    win_rate = float(win_rate_text.strip('%')) / 100
                    if win_rate > 0.7:
                        cell.set_facecolor('#FFCCCC')
                    elif win_rate < 0.3:
                        cell.set_facecolor('#CCCCFF')
            except ValueError:
                pass

    plt.suptitle("expected win rates at nash equilibrium", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    win_probabilities = get_win_probabilities()

    matchups = [
        {
            'team_a': ['zapdos', 'urshifu', 'sylveon'],
            'team_b': ['primarina', 'rillaboom', 'heatran'],
            'output_prefix': 'matchup1'
        },
        {
            'team_a': ['urshifu', 'heatran', 'sylveon'],
            'team_b': ['urshifu', 'zapdos', 'primarina'],
            'output_prefix': 'matchup2'
        },
        {
            'team_a': ['sylveon', 'rillaboom', 'heatran'],
            'team_b': ['zapdos', 'primarina', 'urshifu'],
            'output_prefix': 'matchup3'
        },
        {
            'team_a': ['heatran', 'primarina', 'sylveon'],
            'team_b': ['heatran', 'sylveon', 'urshifu'],
            'output_prefix': 'interesting1'
        },
        {
            'team_a': ['heatran', 'sylveon', 'rillaboom'],
            'team_b': ['primarina', 'heatran', 'sylveon'],
            'output_prefix': 'interesting2'
        },
        {
            'team_a': ['urshifu', 'sylveon', 'rillaboom'],
            'team_b': ['heatran', 'zapdos', 'primarina'],
            'output_prefix': 'interesting3'
        },
        {
            'team_a': ['zapdos', 'sylveon', 'heatran'],
            'team_b': ['heatran', 'rillaboom', 'primarina'],
            'output_prefix': 'boring1'
        },
        {
            'team_a': ['rillaboom', 'primarina', 'sylveon'],
            'team_b': ['rillaboom', 'urshifu', 'zapdos'],
            'output_prefix': 'boring2'
        }
    ]

    results = []

    for matchup in matchups:
        result = analyze_matchup(
            matchup['team_a'],
            matchup['team_b'],
            win_probabilities,
            matchup['output_prefix']
        )
        results.append(result)

    create_summary_visualization(results, "out/pokemon_nash_summary.png")
    create_expected_win_rate_table(results, "out/pokemon_win_rates_table.png")

    print("\nall visualizations have been generated.")
    print("\nsummary of matchups:")
    for i, result in enumerate(results):
        is_mixed = is_mixed_strategy(result['p1_strategy']) or is_mixed_strategy(result['p2_strategy'])
        strategy_type = "mixed strategy" if is_mixed else "pure strategy"
        print(f"{i+1}. {result['team_a']} vs {result['team_b']}: game value = {result['game_value']:.3f} ({strategy_type})")
        print(f"   team a expected win rate: {result['game_value']:.1%}")
        print(f"   team b expected win rate: {(1-result['game_value']):.1%}")

        print(f"   individual pokémon expected win rates:")
        print(f"   team a:")
        for j, pokemon in enumerate(result['team_a']):
            if result['p1_strategy'][j] > 0.01:
                if result['team_a_win_rates'][j] is not None:
                    print(f"     {pokemon}: {result['team_a_win_rates'][j]:.1%}")
                else:
                    print(f"     {pokemon}: not used")

        print(f"   team b:")
        for j, pokemon in enumerate(result['team_b']):
            if result['p2_strategy'][j] > 0.01:
                if result['team_b_win_rates'][j] is not None:
                    print(f"     {pokemon}: {result['team_b_win_rates'][j]:.1%}")
                else:
                    print(f"     {pokemon}: not used")
        print()

if __name__ == "__main__":
    main()
