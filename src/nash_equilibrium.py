import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import matplotlib.tri as mtri
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.colors import LinearSegmentedColormap
import itertools
from src.pokemon_data import get_win_probabilities

class arrow_3d(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
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

def plot_3d_simplex(team_a, team_b, payoff_matrix, p1_strategy, p2_strategy, game_value, output_filename):
    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0.5, np.sqrt(3)/2, 0])

    for ax, strategy, team, title in [(ax1, p1_strategy, team_a, 'Team A Strategy'),
                                     (ax2, p2_strategy, team_b, 'Team B Strategy')]:
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-', linewidth=2)
        ax.plot([v2[0], v3[0]], [v2[1], v3[1]], [v2[2], v3[2]], 'k-', linewidth=2)
        ax.plot([v3[0], v1[0]], [v3[1], v1[1]], [v3[2], v1[2]], 'k-', linewidth=2)

        ax.text(v1[0]-0.1, v1[1]-0.1, v1[2], team[0], fontsize=12)
        ax.text(v2[0]+0.1, v2[1]-0.1, v2[2], team[1], fontsize=12)
        ax.text(v3[0], v3[1]+0.1, v3[2], team[2], fontsize=12)

        nash_pos = strategy[0] * v1 + strategy[1] * v2 + strategy[2] * v3

        ax.scatter([nash_pos[0]], [nash_pos[1]], [nash_pos[2]], color='red', s=100,
                   label=f'Nash Equilibrium\n({strategy[0]:.2f}, {strategy[1]:.2f}, {strategy[2]:.2f})')

        ax.set_title(f'{title}\nOptimal Mixed Strategy', fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

        ax.view_init(elev=45, azim=30)

    plt.suptitle(f"3v3 Pokémon Battle Nash Equilibrium\nGame Value: {game_value:.3f}", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

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

def analyze_3v3_matchup(team_a, team_b, win_probabilities, output_prefix):
    print(f"\nanalyzing {team_a} vs {team_b}")

    payoff_matrix = create_payoff_matrix(team_a, team_b, win_probabilities)
    print(f"payoff matrix:\n{payoff_matrix}")

    p1_strategy, p2_strategy, game_value = compute_nash_equilibrium(payoff_matrix)

    print(f"team a optimal strategy: {p1_strategy}")
    print(f"team b optimal strategy: {p2_strategy}")
    print(f"game value (team a's expected win rate): {game_value:.3f}")

    plot_3d_simplex(team_a, team_b, payoff_matrix, p1_strategy, p2_strategy, game_value,
                   f"out/{output_prefix}_simplex.png")

    plot_heatmap(team_a, team_b, payoff_matrix, f"out/{output_prefix}_heatmap.png")

    return p1_strategy, p2_strategy, game_value

def main():
    win_probabilities = get_win_probabilities()

    matchups = [
        {
            'team_a': ['heatran', 'primarina', 'sylveon'],
            'team_b': ['heatran', 'urshifu', 'sylveon'],
            'output_prefix': 'matchup0'
        },
        {
            'team_a': ['heatran', 'sylveon', 'zapdos'],
            'team_b': ['heatran', 'garchomp', 'urshifu'],
            'output_prefix': 'matchup1'
        },
        {
            'team_a': ['sylveon', 'rillaboom', 'garchomp'],
            'team_b': ['primarina', 'urshifu', 'heatran'],
            'output_prefix': 'matchup2'
        },
        {
            'team_a': ['heatran', 'sylveon', 'primarina'],
            'team_b': ['urshifu', 'rillaboom', 'garchomp'],
            'output_prefix': 'matchup3'
        },
        {
            'team_a': ['heatran', 'sylveon', 'garchomp'],
            'team_b': ['rillaboom', 'primarina', 'urshifu'],
            'output_prefix': 'matchup4'
        },
        {
            'team_a': ['garchomp', 'primarina', 'rillaboom'],
            'team_b': ['heatran', 'urshifu', 'sylveon'],
            'output_prefix': 'matchup5'
        }
    ]

    results = []

    for matchup in matchups:
        p1_strategy, p2_strategy, game_value = analyze_3v3_matchup(
            matchup['team_a'],
            matchup['team_b'],
            win_probabilities,
            matchup['output_prefix']
        )

        results.append({
            'team_a': matchup['team_a'],
            'team_b': matchup['team_b'],
            'p1_strategy': p1_strategy,
            'p2_strategy': p2_strategy,
            'game_value': game_value
        })

    fig, axes = plt.subplots(6, 2, figsize=(20, 15))

    for i, result in enumerate(results):
        team_a = result['team_a']
        team_b = result['team_b']
        p1_strategy = result['p1_strategy']
        p2_strategy = result['p2_strategy']

        axes[i, 0].pie(p1_strategy, labels=team_a, autopct='%1.1f%%',
                       startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
        axes[i, 0].set_title(f"Matchup {i+1}: Team A Strategy")

        axes[i, 1].pie(p2_strategy, labels=team_b, autopct='%1.1f%%',
                       startangle=90, colors=['#FFCC99', '#C2C2F0', '#FFFF99'])
        axes[i, 1].set_title(f"Matchup {i+1}: Team B Strategy")

    plt.suptitle("Nash Equilibrium Strategies for 3v3 Pokémon Battles", fontsize=20)
    plt.tight_layout()
    plt.savefig("out/pokemon_nash_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nvisualization complete! files saved in out/:")
    for matchup in matchups:
        print(f"- {matchup['output_prefix']}_simplex.png")
        print(f"- {matchup['output_prefix']}_heatmap.png")
    print("- pokemon_nash_summary.png")

if __name__ == "__main__":
    main()
