import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# Arrow3D class for 3D visualizations
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

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

def calculate_expected_win_rates(payoff_matrix, p1_strategy, p2_strategy):
    """
    Calculate the expected win rate for each Pokémon when used at equilibrium.

    Returns:
    - team_a_win_rates: List of expected win rates for each Pokémon in Team A
    - team_b_win_rates: List of expected win rates for each Pokémon in Team B
    """
    num_rows, num_cols = payoff_matrix.shape

    # Calculate the expected win rate for each Pokémon in Team A against the mixed strategy of Team B
    team_a_win_rates = []
    for i in range(num_rows):
        # Skip calculation if this Pokémon is never used (to avoid division by zero)
        if p1_strategy[i] < 1e-5:  # Using a small threshold
            team_a_win_rates.append(None)  # None indicates "not used"
            continue

        # Expected win rate against mixed strategy of Team B
        win_rate = 0
        for j in range(num_cols):
            win_rate += payoff_matrix[i, j] * p2_strategy[j]
        team_a_win_rates.append(win_rate)

    # Calculate the expected win rate for each Pokémon in Team B against the mixed strategy of Team A
    team_b_win_rates = []
    for j in range(num_cols):
        # Skip calculation if this Pokémon is never used
        if p2_strategy[j] < 1e-5:
            team_b_win_rates.append(None)
            continue

        # Expected win rate against mixed strategy of Team A (as 1 - Team A's win rate)
        win_rate = 0
        for i in range(num_rows):
            win_rate += (1 - payoff_matrix[i, j]) * p1_strategy[i]
        team_b_win_rates.append(win_rate)

    return team_a_win_rates, team_b_win_rates

def plot_heatmap(team_a, team_b, payoff_matrix, output_filename):
    """Plot a heatmap of the payoff matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a custom colormap: blue (0) -> white (0.5) -> red (1)
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

    # Add text annotations
    for i in range(len(team_a)):
        for j in range(len(team_b)):
            text_color = 'white' if payoff_matrix[i, j] < 0.3 or payoff_matrix[i, j] > 0.7 else 'black'
            ax.text(j, i, f"{payoff_matrix[i, j]:.3f}", ha="center", va="center", color=text_color)

    # Add title
    ax.set_title(f"Battle Matchup Win Probabilities\nTeam A (rows) vs Team B (columns)")

    fig.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_simplex(team_a, team_b, p1_strategy, p2_strategy, game_value, output_filename):
    """Plot the simplex visualization of the Nash equilibrium."""
    # Create a figure with two 3D subplots side by side
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Define the simplex vertices
    v1 = np.array([0, 0, 0])  # Origin
    v2 = np.array([1, 0, 0])  # (1,0,0)
    v3 = np.array([0.5, np.sqrt(3)/2, 0])  # (0.5, sqrt(3)/2, 0)

    # Draw the simplex edges for both teams
    for ax in [ax1, ax2]:
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-', linewidth=2)
        ax.plot([v2[0], v3[0]], [v2[1], v3[1]], [v2[2], v3[2]], 'k-', linewidth=2)
        ax.plot([v3[0], v1[0]], [v3[1], v1[1]], [v3[2], v1[2]], 'k-', linewidth=2)

    # Calculate Nash equilibrium position on the simplex for team A
    nash_pos_a = p1_strategy[0] * v1 + p1_strategy[1] * v2 + p1_strategy[2] * v3

    # Calculate Nash equilibrium position on the simplex for team B
    nash_pos_b = p2_strategy[0] * v1 + p2_strategy[1] * v2 + p2_strategy[2] * v3

    # Plot the Nash equilibrium points
    ax1.scatter([nash_pos_a[0]], [nash_pos_a[1]], [nash_pos_a[2]], color='red', s=100)
    ax2.scatter([nash_pos_b[0]], [nash_pos_b[1]], [nash_pos_b[2]], color='red', s=100)

    # Label the vertices with Pokémon names
    ax1.text(v1[0]-0.1, v1[1]-0.1, v1[2], team_a[0], fontsize=12)
    ax1.text(v2[0]+0.1, v2[1]-0.1, v2[2], team_a[1], fontsize=12)
    ax1.text(v3[0], v3[1]+0.1, v3[2], team_a[2], fontsize=12)

    ax2.text(v1[0]-0.1, v1[1]-0.1, v1[2], team_b[0], fontsize=12)
    ax2.text(v2[0]+0.1, v2[1]-0.1, v2[2], team_b[1], fontsize=12)
    ax2.text(v3[0], v3[1]+0.1, v3[2], team_b[2], fontsize=12)

    # Add legends to show the strategies
    ax1.legend([f"Nash Equilibrium\n({p1_strategy[0]:.2f}, {p1_strategy[1]:.2f}, {p1_strategy[2]:.2f})"],
              loc='lower center', bbox_to_anchor=(0.5, -0.15))
    ax2.legend([f"Nash Equilibrium\n({p2_strategy[0]:.2f}, {p2_strategy[1]:.2f}, {p2_strategy[2]:.2f})"],
              loc='lower center', bbox_to_anchor=(0.5, -0.15))

    # Set titles and adjust the view
    ax1.set_title(f"Team A Strategy\nOptimal Mixed Strategy", fontsize=14)
    ax2.set_title(f"Team B Strategy\nOptimal Mixed Strategy", fontsize=14)

    # Set view angles
    ax1.view_init(elev=30, azim=45)
    ax2.view_init(elev=30, azim=45)

    # Remove axis ticks and labels
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Set limits to ensure the simplex is visible
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 0.1)

    # Add main title
    fig.suptitle(f"3v3 Pokémon Battle Nash Equilibrium\nGame Value: {game_value:.3f}", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_strategies_pie(team_a, team_b, p1_strategy, p2_strategy, game_value, output_filename, team_a_win_rates=None, team_b_win_rates=None):
    """Plot pie charts showing the optimal mixed strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Prepare labels with win rates if provided
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

    # Team A strategy pie chart
    ax1.pie(p1_strategy, labels=a_labels, autopct='%1.1f%%',
           startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
    ax1.set_title(f"Team A Strategy")

    # Team B strategy pie chart
    ax2.pie(p2_strategy, labels=b_labels, autopct='%1.1f%%',
           startangle=90, colors=['#FFCC99', '#C2C2F0', '#FFFF99'])
    ax2.set_title(f"Team B Strategy")

    plt.suptitle(f"Nash Equilibrium Strategies (Game Value: {game_value:.3f})", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def analyze_matchup(team_a, team_b, win_probabilities, output_prefix):
    """Analyze a Pokémon matchup and generate all visualizations."""
    print(f"\nAnalyzing {team_a} vs {team_b}")

    # Create payoff matrix
    payoff_matrix = create_payoff_matrix(team_a, team_b, win_probabilities)
    print(f"Payoff Matrix:\n{payoff_matrix}")

    # Compute Nash equilibrium
    p1_strategy, p2_strategy, game_value = compute_nash_equilibrium(payoff_matrix)

    # Calculate expected win rates
    team_a_win_rates, team_b_win_rates = calculate_expected_win_rates(payoff_matrix, p1_strategy, p2_strategy)

    print(f"Team A strategy: {p1_strategy}")
    print(f"Team B strategy: {p2_strategy}")
    print(f"Game value: {game_value:.3f}")

    # Print expected win rates
    print("Expected win rates for Team A Pokémon:")
    for i, pokemon in enumerate(team_a):
        if team_a_win_rates[i] is not None:
            print(f"  {pokemon}: {team_a_win_rates[i]:.1%}")
        else:
            print(f"  {pokemon}: Not used")

    print("Expected win rates for Team B Pokémon:")
    for i, pokemon in enumerate(team_b):
        if team_b_win_rates[i] is not None:
            print(f"  {pokemon}: {team_b_win_rates[i]:.1%}")
        else:
            print(f"  {pokemon}: Not used")

    # Generate visualizations
    plot_heatmap(team_a, team_b, payoff_matrix, f"{output_prefix}_heatmap.png")
    plot_simplex(team_a, team_b, p1_strategy, p2_strategy, game_value, f"{output_prefix}_simplex.png")
    plot_strategies_pie(team_a, team_b, p1_strategy, p2_strategy, game_value, f"{output_prefix}_strategies.png", team_a_win_rates, team_b_win_rates)

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
    """Check if a strategy is a mixed strategy (more than one non-zero probability)."""
    non_zero = [prob for prob in strategy if prob > threshold]
    return len(non_zero) > 1

def is_interesting_matchup(p1_strategy, p2_strategy, game_value, threshold=0.01):
    """Determine if a matchup is 'interesting' based on Nash equilibrium properties."""
    is_mixed = is_mixed_strategy(p1_strategy, threshold) or is_mixed_strategy(p2_strategy, threshold)
    is_balanced = 0.2 <= game_value <= 0.8
    return is_mixed or is_balanced

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

def create_summary_visualization(matchups, output_filename):
    """Create a summary visualization comparing all matchups."""
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

        # First column: Text summary
        axes[i, 0].axis('off')
        is_mixed = is_mixed_strategy(p1_strategy) or is_mixed_strategy(p2_strategy)
        strategy_type = "Mixed Strategy" if is_mixed else "Pure Strategy"
        summary_text = (
            f"Matchup {i+1}: {team_a} vs {team_b}\n\n"
            f"Game Value: {game_value:.3f}\n\n"
            f"Equilibrium Type: {strategy_type}\n\n"
            f"Team A Strategy:\n"
        )
        for j, (pokemon, prob) in enumerate(zip(team_a, p1_strategy)):
            summary_text += f"  {pokemon}: {prob:.1%}"
            if team_a_win_rates and team_a_win_rates[j] is not None:
                summary_text += f" (Win rate: {team_a_win_rates[j]:.1%})"
            summary_text += "\n"

        summary_text += f"\nTeam B Strategy:\n"
        for j, (pokemon, prob) in enumerate(zip(team_b, p2_strategy)):
            summary_text += f"  {pokemon}: {prob:.1%}"
            if team_b_win_rates and team_b_win_rates[j] is not None:
                summary_text += f" (Win rate: {team_b_win_rates[j]:.1%})"
            summary_text += "\n"

        axes[i, 0].text(0, 0.5, summary_text, va='center', fontsize=12)

        # Prepare labels with win rates
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

        # Second column: Team A pie chart
        axes[i, 1].pie(p1_strategy, labels=a_labels, autopct='%1.1f%%',
                      startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
        axes[i, 1].set_title(f"Team A Strategy")

        # Third column: Team B pie chart
        axes[i, 2].pie(p2_strategy, labels=b_labels, autopct='%1.1f%%',
                      startangle=90, colors=['#FFCC99', '#C2C2F0', '#FFFF99'])
        axes[i, 2].set_title(f"Team B Strategy")

    plt.suptitle("Nash Equilibrium Strategies for 3v3 Pokémon Battles", fontsize=20)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_expected_win_rate_table(results, output_filename):
    """Create a table visualization showing expected win rates for all matchups."""
    num_matchups = len(results)

    # Count unique Pokémon across all matchups
    all_pokemon = set()
    for result in results:
        all_pokemon.update(result['team_a'])
        all_pokemon.update(result['team_b'])

    all_pokemon = sorted(list(all_pokemon))
    num_pokemon = len(all_pokemon)

    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    # Create the table data
    table_data = []
    header = ["Matchup", "Team", "Game Value"] + all_pokemon
    table_data.append(header)

    for i, result in enumerate(results):
        # Team A row
        row_a = [f"Matchup {i+1}", "Team A", f"{result['game_value']:.3f}"]
        for pokemon in all_pokemon:
            try:
                idx = result['team_a'].index(pokemon)
                win_rate = result['team_a_win_rates'][idx]
                if win_rate is None:
                    row_a.append("Not used")
                else:
                    row_a.append(f"{win_rate:.1%}")
            except ValueError:
                row_a.append("N/A")

        # Team B row
        row_b = [f"Matchup {i+1}", "Team B", f"{1-result['game_value']:.3f}"]
        for pokemon in all_pokemon:
            try:
                idx = result['team_b'].index(pokemon)
                win_rate = result['team_b_win_rates'][idx]
                if win_rate is None:
                    row_b.append("Not used")
                else:
                    row_b.append(f"{win_rate:.1%}")
            except ValueError:
                row_b.append("N/A")

        table_data.append(row_a)
        table_data.append(row_b)

    # Create the table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', cellLoc='center', colColours=['#DDDDDD']*len(header))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight interesting cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold')
        elif col >= 3:  # Pokemon columns
            try:
                win_rate_text = cell.get_text().get_text()
                if win_rate_text not in ["N/A", "Not used"]:
                    win_rate = float(win_rate_text.strip('%')) / 100
                    if win_rate > 0.7:
                        cell.set_facecolor('#FFCCCC')  # Light red for high win rates
                    elif win_rate < 0.3:
                        cell.set_facecolor('#CCCCFF')  # Light blue for low win rates
            except ValueError:
                pass

    plt.suptitle("Expected Win Rates at Nash Equilibrium", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    win_probabilities = get_win_probabilities()

    # Define the matchups to analyze
    matchups = [
        # Original matchups from your paper
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
        },

        # Most interesting matchups from generated data
        {
            'team_a': ['Heatran', 'Primarina', 'Sylveon'],
            'team_b': ['Heatran', 'Sylveon', 'Urshifu'],
            'output_prefix': 'interesting1'
        },
        {
            'team_a': ['Heatran', 'Sylveon', 'Rillaboom'],
            'team_b': ['Primarina', 'Heatran', 'Sylveon'],
            'output_prefix': 'interesting2'
        },
        {
            'team_a': ['Urshifu', 'Sylveon', 'Rillaboom'],
            'team_b': ['Heatran', 'Zapdos', 'Primarina'],
            'output_prefix': 'interesting3'
        },

        # Boring matchups (pure strategies)
        {
            'team_a': ['Zapdos', 'Sylveon', 'Heatran'],
            'team_b': ['Heatran', 'Rillaboom', 'Primarina'],
            'output_prefix': 'boring1'
        },
        {
            'team_a': ['Rillaboom', 'Primarina', 'Sylveon'],
            'team_b': ['Rillaboom', 'Urshifu', 'Zapdos'],
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

    # Create a summary visualization for all matchups
    create_summary_visualization(results, "pokemon_nash_summary.png")

    # Create a table of expected win rates
    create_expected_win_rate_table(results, "pokemon_win_rates_table.png")

    print("\nAll visualizations have been generated.")
    print("\nSummary of matchups:")
    for i, result in enumerate(results):
        is_mixed = is_mixed_strategy(result['p1_strategy']) or is_mixed_strategy(result['p2_strategy'])
        strategy_type = "Mixed Strategy" if is_mixed else "Pure Strategy"
        print(f"{i+1}. {result['team_a']} vs {result['team_b']}: Game value = {result['game_value']:.3f} ({strategy_type})")
        print(f"   Team A expected win rate: {result['game_value']:.1%}")
        print(f"   Team B expected win rate: {(1-result['game_value']):.1%}")

        # Print expected win rates by Pokémon
        print(f"   Individual Pokémon expected win rates:")
        print(f"   Team A:")
        for j, pokemon in enumerate(result['team_a']):
            if result['p1_strategy'][j] > 0.01:  # Only show for Pokémon that are actually used
                if result['team_a_win_rates'][j] is not None:
                    print(f"     {pokemon}: {result['team_a_win_rates'][j]:.1%}")
                else:
                    print(f"     {pokemon}: Not used")

        print(f"   Team B:")
        for j, pokemon in enumerate(result['team_b']):
            if result['p2_strategy'][j] > 0.01:  # Only show for Pokémon that are actually used
                if result['team_b_win_rates'][j] is not None:
                    print(f"     {pokemon}: {result['team_b_win_rates'][j]:.1%}")
                else:
                    print(f"     {pokemon}: Not used")
        print()

if __name__ == "__main__":
    main()
