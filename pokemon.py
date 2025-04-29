import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import matplotlib.tri as mtri
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.colors import LinearSegmentedColormap
import itertools

# class to create 3D arrows for the visualization
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# given matchup win probabilities
def create_payoff_matrix(team_a, team_b, win_probabilities):
    """
    Create a payoff matrix for 3v3 Pokémon battles.

    Args:
        team_a: List of Pokémon names for Team A
        team_b: List of Pokémon names for Team B
        win_probabilities: Dictionary mapping matchup tuples to win probabilities

    Returns:
        Payoff matrix where A[i,j] is the win probability when Team A's Pokémon i
        faces Team B's Pokémon j
    """
    n_a = len(team_a)
    n_b = len(team_b)
    payoff_matrix = np.zeros((n_a, n_b))

    for i, pokemon_a in enumerate(team_a):
        for j, pokemon_b in enumerate(team_b):
            matchup = (pokemon_a, pokemon_b)
            # default to 50/50 if matchup not found
            payoff_matrix[i, j] = win_probabilities.get(matchup, 0.5)

    return payoff_matrix

def compute_nash_equilibrium(payoff_matrix):
    """
    Compute the Nash equilibrium for a zero-sum game using linear programming.

    Args:
        payoff_matrix: A numpy array where A[i,j] is the payoff when player 1
                      plays strategy i and player 2 plays strategy j

    Returns:
        tuple (p1_strategy, p2_strategy, value) where:
        - p1_strategy is player 1's mixed strategy
        - p2_strategy is player 2's mixed strategy
        - value is the value of the game to player 1
    """
    num_rows, num_cols = payoff_matrix.shape

    # --- Player 1 (Row Player) --- #

    c = np.zeros(num_rows + 1)
    c[-1] = -1  # maximize v (which is the last variable)

    # constraints for player 1's linear program
    A_ub = np.zeros((num_cols, num_rows + 1))
    for j in range(num_cols):
        A_ub[j, :-1] = -payoff_matrix[:, j]
        A_ub[j, -1] = 1
    b_ub = np.zeros(num_cols)

    # constraint that probabilities sum to 1
    A_eq = np.zeros((1, num_rows + 1))
    A_eq[0, :-1] = 1
    b_eq = np.ones(1)

    bounds = [(0, 1) for _ in range(num_rows)] + [(None, None)]

    # solve!
    res1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # extract player 1's strategy and value
    p1_strategy = res1.x[:-1]
    v = -res1.fun

    # --- Player 2 (Column Player) --- #

    c = np.zeros(num_cols + 1)
    c[-1] = 1  # minimize u (which is the last variable)

    # constraints for player 2's linear program
    A_ub = np.zeros((num_rows, num_cols + 1))
    for i in range(num_rows):
        A_ub[i, :-1] = payoff_matrix[i, :]
        A_ub[i, -1] = -1
    b_ub = np.zeros(num_rows)

    # constraint that probabilities sum to 1
    A_eq = np.zeros((1, num_cols + 1))
    A_eq[0, :-1] = 1
    b_eq = np.ones(1)

    # bounds for variables
    bounds = [(0, 1) for _ in range(num_cols)] + [(None, None)]

    # solve!
    res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # extract player 2's strategy and value
    p2_strategy = res2.x[:-1]
    u = res2.fun

    # check if values match (they should be close)
    if not np.isclose(u, v, atol=1e-5):
        print(f"Warning: Game values do not match. v = {v}, u = {u}")

    return p1_strategy, p2_strategy, v

def plot_3d_simplex(team_a, team_b, payoff_matrix, p1_strategy, p2_strategy, game_value, output_filename):
    """
    Visualize the Nash equilibrium on a 3D simplex.

    Args:
        team_a: List of Pokémon names for Team A
        team_b: List of Pokémon names for Team B
        payoff_matrix: The payoff matrix
        p1_strategy: Player 1's mixed strategy
        p2_strategy: Player 2's mixed strategy
        game_value: The value of the game
        output_filename: Filename to save the visualization
    """
    fig = plt.figure(figsize=(18, 8))

    # create two subplots - one for each team's strategy
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # coordinates for simplex in 3D
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0.5, np.sqrt(3)/2, 0])

    # plot simplex for Team A
    for ax, strategy, team, title in [(ax1, p1_strategy, team_a, 'Team A Strategy'),
                                     (ax2, p2_strategy, team_b, 'Team B Strategy')]:
        # Plot the simplex
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-', linewidth=2)
        ax.plot([v2[0], v3[0]], [v2[1], v3[1]], [v2[2], v3[2]], 'k-', linewidth=2)
        ax.plot([v3[0], v1[0]], [v3[1], v1[1]], [v3[2], v1[2]], 'k-', linewidth=2)

        # Add labels for each vertex (Pokémon)
        ax.text(v1[0]-0.1, v1[1]-0.1, v1[2], team[0], fontsize=12)
        ax.text(v2[0]+0.1, v2[1]-0.1, v2[2], team[1], fontsize=12)
        ax.text(v3[0], v3[1]+0.1, v3[2], team[2], fontsize=12)

        # Calculate the position of the Nash equilibrium in the simplex
        nash_pos = strategy[0] * v1 + strategy[1] * v2 + strategy[2] * v3

        # Plot the Nash equilibrium point
        ax.scatter([nash_pos[0]], [nash_pos[1]], [nash_pos[2]], color='red', s=100,
                   label=f'Nash Equilibrium\n({strategy[0]:.2f}, {strategy[1]:.2f}, {strategy[2]:.2f})')

        # Add title and settings
        ax.set_title(f'{title}\nOptimal Mixed Strategy', fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

        # Improve visualization by setting view angle
        ax.view_init(elev=45, azim=30)

    # Add overall title with game value
    plt.suptitle(f"3v3 Pokémon Battle Nash Equilibrium\nGame Value: {game_value:.3f}", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return

def plot_heatmap(team_a, team_b, payoff_matrix, output_filename):
    """
    Plot a heatmap of the payoff matrix.

    Args:
        team_a: List of Pokémon names for Team A
        team_b: List of Pokémon names for Team B
        payoff_matrix: The payoff matrix
        output_filename: Filename to save the visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a custom colormap that goes from blue (0) to white (0.5) to red (1)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list('BuWtRd', colors, N=100)

    # Plot the heatmap
    im = ax.imshow(payoff_matrix, cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Win Probability for Team A')

    # Add labels and ticks
    ax.set_xticks(np.arange(len(team_b)))
    ax.set_yticks(np.arange(len(team_a)))
    ax.set_xticklabels(team_b)
    ax.set_yticklabels(team_a)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorized text annotations in each cell
    for i in range(len(team_a)):
        for j in range(len(team_b)):
            text_color = 'white' if payoff_matrix[i, j] < 0.3 or payoff_matrix[i, j] > 0.7 else 'black'
            ax.text(j, i, f"{payoff_matrix[i, j]:.3f}", ha="center", va="center", color=text_color)

    # Add title
    ax.set_title(f"Battle Matchup Win Probabilities\nTeam A (rows) vs Team B (columns)")

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

    return

def analyze_3v3_matchup(team_a, team_b, win_probabilities, output_prefix):
    """
    Analyze a 3v3 Pokémon matchup, find Nash equilibrium, and create visualizations.

    Args:
        team_a: List of Pokémon names for Team A
        team_b: List of Pokémon names for Team B
        win_probabilities: Dictionary mapping matchup tuples to win probabilities
        output_prefix: Prefix for output filenames
    """
    print(f"\nAnalyzing {team_a} vs {team_b}")

    # Create payoff matrix
    payoff_matrix = create_payoff_matrix(team_a, team_b, win_probabilities)
    print(f"Payoff Matrix:\n{payoff_matrix}")

    # Compute Nash equilibrium
    p1_strategy, p2_strategy, game_value = compute_nash_equilibrium(payoff_matrix)

    print(f"Team A optimal strategy: {p1_strategy}")
    print(f"Team B optimal strategy: {p2_strategy}")
    print(f"Game value (Team A's expected win rate): {game_value:.3f}")

    # Plot the Nash equilibrium on a simplex
    plot_3d_simplex(team_a, team_b, payoff_matrix, p1_strategy, p2_strategy, game_value,
                   f"{output_prefix}_simplex.png")

    # Plot the payoff matrix as a heatmap
    plot_heatmap(team_a, team_b, payoff_matrix, f"{output_prefix}_heatmap.png")

    return p1_strategy, p2_strategy, game_value

# Define the matchup win probabilities based on your paper
def get_win_probabilities():
    """Return a dictionary of matchup win probabilities."""
    probabilities = {
        # Format: (pokemon_a, pokemon_b): probability of pokemon_a winning
        ('Primarina', 'Primarina'): 0.5,
        ('Primarina', 'Sylveon'): 0.09583,
        ('Primarina', 'Rillaboom'): 0.0252,
        ('Primarina', 'Heatran'): 1.0,
        ('Primarina', 'Urshifu'): 1.0,
        ('Primarina', 'Zapdos'): 0.0,  # Zapdos wins 100% against Primarina

        ('Sylveon', 'Primarina'): 0.90417,
        ('Sylveon', 'Sylveon'): 0.5,
        ('Sylveon', 'Rillaboom'): 0.387,
        ('Sylveon', 'Heatran'): 0.0,
        ('Sylveon', 'Urshifu'): 1.0,
        ('Sylveon', 'Zapdos'): 0.00625,  # Zapdos wins 99.375% against Sylveon

        ('Rillaboom', 'Primarina'): 0.9748,
        ('Rillaboom', 'Sylveon'): 0.613,
        ('Rillaboom', 'Rillaboom'): 0.5,
        ('Rillaboom', 'Heatran'): 0.95,
        ('Rillaboom', 'Urshifu'): 0.6954,
        ('Rillaboom', 'Zapdos'): 0.0165,  # Zapdos wins 98.35% against Rillaboom

        ('Heatran', 'Primarina'): 0.0,
        ('Heatran', 'Sylveon'): 1.0,
        ('Heatran', 'Rillaboom'): 0.05,
        ('Heatran', 'Heatran'): 0.5,
        ('Heatran', 'Urshifu'): 0.0,
        ('Heatran', 'Zapdos'): 0.0,  # Zapdos wins 100% against Heatran

        ('Urshifu', 'Primarina'): 0.0,
        ('Urshifu', 'Sylveon'): 0.0,
        ('Urshifu', 'Rillaboom'): 0.3046,
        ('Urshifu', 'Heatran'): 1.0,
        ('Urshifu', 'Urshifu'): 0.5,
        ('Urshifu', 'Zapdos'): 0.3,  # Zapdos wins 70% against Urshifu

        ('Zapdos', 'Primarina'): 1.0,
        ('Zapdos', 'Sylveon'): 0.99375,
        ('Zapdos', 'Rillaboom'): 0.9835,
        ('Zapdos', 'Heatran'): 1.0,
        ('Zapdos', 'Urshifu'): 0.7,
        ('Zapdos', 'Zapdos'): 0.5
    }
    return probabilities

# Generate visualizations for the three matchups
def main():
    # Get win probabilities
    win_probabilities = get_win_probabilities()

    # Define the three matchups from your request
    matchups = [
        {
            'team_a': ['Zapdos', 'Urshifu', 'Sylveon'],
            'team_b': ['Primarina', 'Rillaboom', 'Heatran'],
            'output_prefix': 'matchup1'
        },
        {
            'team_a': ['Urshifu', 'Heatran', 'Sylveon'],
            'team_b': ['Urshifu', 'Zapdos', 'Primarina'],
            'output_prefix': 'matchup2'
        },
        {
            'team_a': ['Sylveon', 'Rillaboom', 'Heatran'],
            'team_b': ['Zapdos', 'Primarina', 'Urshifu'],
            'output_prefix': 'matchup3'
        }
    ]

    results = []

    # Analyze each matchup
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

    # Create a summary visualization for all matchups
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))

    for i, result in enumerate(results):
        team_a = result['team_a']
        team_b = result['team_b']
        p1_strategy = result['p1_strategy']
        p2_strategy = result['p2_strategy']

        # Team A strategy pie chart
        axes[i, 0].pie(p1_strategy, labels=team_a, autopct='%1.1f%%',
                       startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
        axes[i, 0].set_title(f"Matchup {i+1}: Team A Strategy")

        # Team B strategy pie chart
        axes[i, 1].pie(p2_strategy, labels=team_b, autopct='%1.1f%%',
                       startangle=90, colors=['#FFCC99', '#C2C2F0', '#FFFF99'])
        axes[i, 1].set_title(f"Matchup {i+1}: Team B Strategy")

    plt.suptitle("Nash Equilibrium Strategies for 3v3 Pokémon Battles", fontsize=20)
    plt.tight_layout()
    plt.savefig("pokemon_nash_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nVisualization complete! Files saved:")
    for matchup in matchups:
        print(f"- {matchup['output_prefix']}_simplex.png")
        print(f"- {matchup['output_prefix']}_heatmap.png")
    print("- pokemon_nash_summary.png")

if __name__ == "__main__":
    main()
