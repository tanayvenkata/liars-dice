#!/usr/bin/env python3
"""
Visualization script for Liar's Dice tournament.

Generates:
1. Round Robin Heatmap - 6x6 win rate matrix
2. Tournament Bracket - Single elimination seeded by round robin
3. V3 Adaptation Charts - Threshold evolution and win rate over time
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

# Try to import seaborn, fall back to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed, using matplotlib for heatmap")

from agents.deterministic_agent import create_personality_agent, PERSONALITIES
from game.engine import GameEngine
from stats_utils import margin_of_error, format_win_rate_with_ci

# Default sample sizes - increased for statistical power
DEFAULT_GAMES_PER_MATCHUP = 10000
ROLLING_WINDOW_SIZE = 100  # Increased from 50 for smoother curves

# Personality order for consistent visualization
PERSONALITY_ORDER = ["MANIAC", "AGGRESSIVE", "PASSIVE", "CALLING_STATION", "BALANCED", "ADAPTIVE"]


def play_game_silent(engine, agents):
    """Play a single game silently, return winner index."""
    engine.start_game()
    while not engine.is_game_over:
        current = engine.state.current_player
        view = engine.get_player_view(current)
        action = agents[current].get_action(view)
        result = engine.apply_action(current, action)
        if result:
            for agent in agents:
                agent.notify_round_result(result)
    return engine.state.winner


def run_matchup(p1_type, p2_type, num_games=None, verbose=True):
    """Run a head-to-head matchup between two personality types.

    IMPORTANT: ADAPTIVE agents are now persistent across games to enable learning.
    Other agent types are still reset each game (they're stateless anyway).

    Returns:
        tuple: (win_rate, wins, num_games) for CI calculation
    """
    if num_games is None:
        num_games = DEFAULT_GAMES_PER_MATCHUP

    wins = [0, 0]

    # Create ADAPTIVE agents once and reuse (persistent memory)
    # Other agents can be recreated (they're stateless)
    p1_is_adaptive = p1_type == "ADAPTIVE"
    p2_is_adaptive = p2_type == "ADAPTIVE"

    p1_persistent = create_personality_agent(p1_type, quiet=True) if p1_is_adaptive else None
    p2_persistent = create_personality_agent(p2_type, quiet=True) if p2_is_adaptive else None

    for game_num in range(num_games):
        # Use persistent agent for ADAPTIVE, fresh agent for others
        agent1 = p1_persistent if p1_is_adaptive else create_personality_agent(p1_type, quiet=True)
        agent2 = p2_persistent if p2_is_adaptive else create_personality_agent(p2_type, quiet=True)
        agents = [agent1, agent2]

        engine = GameEngine(starting_dice=5)
        engine.initialize_game(num_players=2)
        winner = play_game_silent(engine, agents)
        wins[winner] += 1

        if verbose and (game_num + 1) % 1000 == 0:
            print(f"  {p1_type} vs {p2_type}: {game_num + 1}/{num_games} games...")

    return wins[0] / num_games, wins[0], num_games


def run_round_robin(games_per_matchup=None):
    """Run all pairwise matchups, return 6x6 win rate matrix and raw data.

    Returns:
        tuple: (matrix, wins_matrix, games_matrix)
            - matrix: win rate matrix[i][j] = P1 win rate when row plays col
            - wins_matrix: raw wins for CI calculation
            - games_matrix: number of games played
    """
    if games_per_matchup is None:
        games_per_matchup = DEFAULT_GAMES_PER_MATCHUP

    n = len(PERSONALITY_ORDER)
    matrix = np.zeros((n, n))
    wins_matrix = np.zeros((n, n), dtype=int)
    games_matrix = np.zeros((n, n), dtype=int)

    print(f"\nRunning Round Robin ({games_per_matchup} games per matchup)")
    print("=" * 60)

    total_matchups = n * (n - 1) // 2
    matchup_num = 0

    for i, p1 in enumerate(PERSONALITY_ORDER):
        for j, p2 in enumerate(PERSONALITY_ORDER):
            if i == j:
                matrix[i][j] = 0.5  # Self-play is 50%
                wins_matrix[i][j] = games_per_matchup // 2
                games_matrix[i][j] = games_per_matchup
            elif i < j:
                matchup_num += 1
                print(f"\n[{matchup_num}/{total_matchups}] {p1} vs {p2}")
                win_rate, wins, n_games = run_matchup(p1, p2, games_per_matchup, verbose=True)
                matrix[i][j] = win_rate
                matrix[j][i] = 1 - win_rate  # Symmetric
                wins_matrix[i][j] = wins
                wins_matrix[j][i] = n_games - wins
                games_matrix[i][j] = n_games
                games_matrix[j][i] = n_games
                moe = margin_of_error(wins, n_games)
                print(f"  Result: {p1} {win_rate:.1%} ± {moe:.1%} - {1-win_rate:.1%} {p2}")

    return matrix, wins_matrix, games_matrix


def plot_heatmap(matrix, labels, filename, wins_matrix=None, games_matrix=None):
    """Generate heatmap visualization of win rates with optional CIs."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Expanded color scale to show more extreme values
    vmin, vmax = 0.25, 0.75

    if HAS_SEABORN:
        # Use seaborn for nicer heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.0%',
            cmap='RdYlGn',
            center=0.5,
            vmin=vmin,
            vmax=vmax,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Win Rate'}
        )
    else:
        # Fallback to matplotlib
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax)

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = f'{matrix[i, j]:.0%}'
                ax.text(j, i, text, ha='center', va='center', fontsize=10)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        plt.colorbar(im, ax=ax, label='Win Rate')

    # Add sample size info to title
    n_games = games_matrix[0, 1] if games_matrix is not None else DEFAULT_GAMES_PER_MATCHUP
    moe = margin_of_error(int(n_games * 0.5), n_games) if n_games > 0 else 0
    ax.set_title(f'Round Robin Results (n={n_games:,} games/matchup, ±{moe:.1%} margin)\n(Row vs Column Win Rate)',
                 fontsize=14)
    ax.set_xlabel('Opponent')
    ax.set_ylabel('Agent')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def get_round_robin_standings(matrix):
    """Calculate standings from round robin matrix.

    Returns:
        list: [(personality, total_wins, avg_win_rate), ...] sorted by performance
    """
    standings = []
    n = len(PERSONALITY_ORDER)

    for i, personality in enumerate(PERSONALITY_ORDER):
        # Sum wins against all opponents (excluding self)
        total_win_rate = sum(matrix[i][j] for j in range(n) if i != j) / (n - 1)
        standings.append((personality, total_win_rate))

    # Sort by win rate descending
    standings.sort(key=lambda x: x[1], reverse=True)
    return standings


def run_bracket_tournament(seeding, games_per_round=None):
    """Run single elimination tournament based on seeding.

    Args:
        seeding: List of personality types in seed order (1st seed first)
        games_per_round: Games per matchup (defaults to DEFAULT_GAMES_PER_MATCHUP)

    Returns:
        dict: Tournament results with bracket progression
    """
    if games_per_round is None:
        games_per_round = DEFAULT_GAMES_PER_MATCHUP
    results = {
        'seeding': seeding,
        'rounds': [],
        'champion': None
    }

    # Bracket: 1v6, 2v5, 3v4 for quarterfinals equivalent
    # Then winners play semis, finals

    print("\n" + "=" * 60)
    print("TOURNAMENT BRACKET")
    print("=" * 60)

    # Round 1: Top 3 seeds get byes if we only have 6 (or play bottom 3)
    # With 6 players: 1v6, 2v5, 3v4
    current_round = []
    matchups = [(0, 5), (1, 4), (2, 3)]  # Seed indices

    print("\n--- Round 1 (Quarterfinals) ---")
    for seed1_idx, seed2_idx in matchups:
        p1 = seeding[seed1_idx]
        p2 = seeding[seed2_idx]
        print(f"\n#{seed1_idx+1} {p1} vs #{seed2_idx+1} {p2}")

        win_rate, _, _ = run_matchup(p1, p2, games_per_round, verbose=False)
        winner = p1 if win_rate > 0.5 else p2
        current_round.append({
            'matchup': (p1, p2),
            'seeds': (seed1_idx+1, seed2_idx+1),
            'win_rate': win_rate if win_rate > 0.5 else 1 - win_rate,
            'winner': winner
        })
        print(f"  Winner: {winner} ({max(win_rate, 1-win_rate):.1%})")

    results['rounds'].append(current_round)
    winners = [m['winner'] for m in current_round]

    # Round 2: Semifinals (winner of 1v6 vs winner of 3v4, winner of 2v5 gets bye or plays)
    # Standard bracket: W(1v6) vs W(3v4), then winner vs W(2v5)
    print("\n--- Round 2 (Semifinals) ---")
    semi_matchups = [(winners[0], winners[2]), (winners[1], None)]  # Second winner gets bye

    current_round = []

    # First semifinal
    p1, p2 = winners[0], winners[2]
    print(f"\n{p1} vs {p2}")
    win_rate, _, _ = run_matchup(p1, p2, games_per_round, verbose=False)
    semi_winner1 = p1 if win_rate > 0.5 else p2
    current_round.append({
        'matchup': (p1, p2),
        'win_rate': win_rate if win_rate > 0.5 else 1 - win_rate,
        'winner': semi_winner1
    })
    print(f"  Winner: {semi_winner1} ({max(win_rate, 1-win_rate):.1%})")

    # Second semifinal (winner of above vs middle bracket winner)
    p1, p2 = semi_winner1, winners[1]
    print(f"\n{p1} vs {p2}")
    win_rate, _, _ = run_matchup(p1, p2, games_per_round, verbose=False)
    finalist1 = p1 if win_rate > 0.5 else p2
    finalist2 = p2 if win_rate > 0.5 else p1  # The other one for 3rd place
    current_round.append({
        'matchup': (p1, p2),
        'win_rate': win_rate if win_rate > 0.5 else 1 - win_rate,
        'winner': finalist1
    })
    print(f"  Winner: {finalist1} ({max(win_rate, 1-win_rate):.1%})")

    results['rounds'].append(current_round)
    results['champion'] = finalist1
    results['runner_up'] = finalist2

    print("\n" + "=" * 60)
    print(f"CHAMPION: {results['champion']}")
    print("=" * 60)

    return results


def plot_bracket(results, filename):
    """Generate bracket visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 95, 'LIAR\'S DICE TOURNAMENT', fontsize=20, ha='center', weight='bold')

    # Seeding info
    seeding = results['seeding']
    y_positions = [75, 60, 45, 30, 15, 0]

    # Round 1 matchups (left side)
    ax.text(5, 85, 'ROUND 1', fontsize=12, weight='bold')

    r1 = results['rounds'][0]
    matchup_y = [70, 50, 30]
    for i, match in enumerate(r1):
        p1, p2 = match['matchup']
        s1, s2 = match['seeds']
        winner = match['winner']
        wr = match['win_rate']

        y = matchup_y[i]
        # Box for matchup
        rect = mpatches.FancyBboxPatch((5, y-5), 25, 12,
                                        boxstyle="round,pad=0.02",
                                        facecolor='lightblue' if winner == p1 else 'white',
                                        edgecolor='black')
        ax.add_patch(rect)
        ax.text(17.5, y+3, f'#{s1} {p1}', ha='center', fontsize=9)

        rect2 = mpatches.FancyBboxPatch((5, y-17), 25, 12,
                                         boxstyle="round,pad=0.02",
                                         facecolor='lightblue' if winner == p2 else 'white',
                                         edgecolor='black')
        ax.add_patch(rect2)
        ax.text(17.5, y-11, f'#{s2} {p2}', ha='center', fontsize=9)

        # Arrow to next round
        ax.annotate('', xy=(35, y-5), xytext=(30, y-5),
                   arrowprops=dict(arrowstyle='->', color='gray'))

    # Semifinals
    ax.text(38, 85, 'SEMIS', fontsize=12, weight='bold')

    if len(results['rounds']) > 1:
        r2 = results['rounds'][1]
        semi_y = [55, 35]
        for i, match in enumerate(r2):
            p1, p2 = match['matchup']
            winner = match['winner']
            y = semi_y[i]

            rect = mpatches.FancyBboxPatch((38, y-5), 25, 12,
                                            boxstyle="round,pad=0.02",
                                            facecolor='lightgreen' if winner == p1 else 'white',
                                            edgecolor='black')
            ax.add_patch(rect)
            ax.text(50.5, y+3, p1, ha='center', fontsize=9)

            rect2 = mpatches.FancyBboxPatch((38, y-17), 25, 12,
                                             boxstyle="round,pad=0.02",
                                             facecolor='lightgreen' if winner == p2 else 'white',
                                             edgecolor='black')
            ax.add_patch(rect2)
            ax.text(50.5, y-11, p2, ha='center', fontsize=9)

    # Champion
    ax.text(70, 85, 'CHAMPION', fontsize=12, weight='bold')
    champion = results.get('champion', '?')
    rect = mpatches.FancyBboxPatch((68, 40), 28, 15,
                                    boxstyle="round,pad=0.02",
                                    facecolor='gold',
                                    edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(82, 47.5, champion, ha='center', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def track_v3_adaptation(opponent_type, num_games=500):
    """Track V3 (ADAPTIVE) threshold changes and win rate over games.

    Returns:
        dict: Tracking data with thresholds and wins per game
    """
    print(f"\nTracking ADAPTIVE vs {opponent_type} ({num_games} games)...")

    # Single persistent ADAPTIVE agent
    adaptive = create_personality_agent("ADAPTIVE", quiet=True)

    data = {
        'games': [],
        'call_threshold': [],
        'raise_threshold': [],
        'cumulative_wins': [],
        'rolling_win_rate': []
    }

    wins = 0
    recent_results = []  # For rolling window

    for game_num in range(num_games):
        # Fresh opponent each game (stateless)
        opponent = create_personality_agent(opponent_type, quiet=True)
        agents = [adaptive, opponent]

        engine = GameEngine(starting_dice=5)
        engine.initialize_game(num_players=2)
        winner = play_game_silent(engine, agents)

        if winner == 0:
            wins += 1
            recent_results.append(1)
        else:
            recent_results.append(0)

        # Keep rolling window of last ROLLING_WINDOW_SIZE
        if len(recent_results) > ROLLING_WINDOW_SIZE:
            recent_results.pop(0)

        # Get current thresholds from ADAPTIVE's opponent model
        model = adaptive._opponent_models.get(1)  # Opponent is player 1
        if model:
            call_th, raise_th = model.get_adjusted_thresholds()
        else:
            call_th, raise_th = 0.25, 0.30  # GTO defaults

        data['games'].append(game_num + 1)
        data['call_threshold'].append(call_th)
        data['raise_threshold'].append(raise_th)
        data['cumulative_wins'].append(wins)
        data['rolling_win_rate'].append(sum(recent_results) / len(recent_results))

        if (game_num + 1) % 100 == 0:
            print(f"  Game {game_num + 1}: Win rate = {wins/(game_num+1):.1%}, "
                  f"call_th = {call_th:.1%}, raise_th = {raise_th:.1%}")

    data['opponent'] = opponent_type
    data['final_win_rate'] = wins / num_games

    return data


def track_v3_vs_v3(num_games=500):
    """Track two V3 agents learning each other simultaneously.

    Returns:
        dict: Tracking data for both agents
    """
    print(f"\nTracking V3 vs V3 ({num_games} games)...")

    from agents.deterministic_agent import DeterministicV3Agent

    # Two persistent V3 agents
    v3_a = DeterministicV3Agent("V3-A", quiet=True, memory_enabled=True)
    v3_b = DeterministicV3Agent("V3-B", quiet=True, memory_enabled=True)

    data = {
        'games': [],
        'a_call_threshold': [],
        'a_raise_threshold': [],
        'b_call_threshold': [],
        'b_raise_threshold': [],
        'a_wins': [],
        'rolling_win_rate': []
    }

    a_wins = 0
    recent_results = []

    for game_num in range(num_games):
        agents = [v3_a, v3_b]

        engine = GameEngine(starting_dice=5)
        engine.initialize_game(num_players=2)

        # Set player IDs
        v3_a._my_id = 0
        v3_b._my_id = 1

        winner = play_game_silent(engine, agents)

        if winner == 0:
            a_wins += 1
            recent_results.append(1)
        else:
            recent_results.append(0)

        if len(recent_results) > ROLLING_WINDOW_SIZE:
            recent_results.pop(0)

        # Get thresholds from each agent's model of the other
        model_a = v3_a._opponent_models.get(1)  # A's model of B
        model_b = v3_b._opponent_models.get(0)  # B's model of A

        a_call, a_raise = model_a.get_adjusted_thresholds() if model_a else (0.25, 0.30)
        b_call, b_raise = model_b.get_adjusted_thresholds() if model_b else (0.25, 0.30)

        data['games'].append(game_num + 1)
        data['a_call_threshold'].append(a_call)
        data['a_raise_threshold'].append(a_raise)
        data['b_call_threshold'].append(b_call)
        data['b_raise_threshold'].append(b_raise)
        data['a_wins'].append(a_wins)
        data['rolling_win_rate'].append(sum(recent_results) / len(recent_results))

        if (game_num + 1) % 100 == 0:
            print(f"  Game {game_num + 1}: V3-A win rate = {a_wins/(game_num+1):.1%}")

    data['final_a_win_rate'] = a_wins / num_games

    # Get final opponent classifications
    model_a = v3_a._opponent_models.get(1)
    model_b = v3_b._opponent_models.get(0)
    data['a_sees_b_as'] = model_a.get_opponent_type().value if model_a else "unknown"
    data['b_sees_a_as'] = model_b.get_opponent_type().value if model_b else "unknown"

    return data


def plot_v3_vs_v3(data, filename):
    """Generate V3 vs V3 adaptation chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    games = data['games']

    # Top plot: Both agents' thresholds
    ax1.plot(games, [t * 100 for t in data['a_call_threshold']],
             label='V3-A Call', color='blue', linewidth=2)
    ax1.plot(games, [t * 100 for t in data['a_raise_threshold']],
             label='V3-A Raise', color='blue', linewidth=2, linestyle='--')
    ax1.plot(games, [t * 100 for t in data['b_call_threshold']],
             label='V3-B Call', color='red', linewidth=2)
    ax1.plot(games, [t * 100 for t in data['b_raise_threshold']],
             label='V3-B Raise', color='red', linewidth=2, linestyle='--')

    # GTO baselines
    ax1.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='GTO Call (25%)')
    ax1.axhline(y=30, color='gray', linestyle=':', alpha=0.5, label='GTO Raise (30%)')

    ax1.set_ylabel('Threshold (%)', fontsize=12)
    ax1.set_title(f'V3 vs V3: Mutual Learning\n'
                  f'(A sees B as: {data["a_sees_b_as"]}, B sees A as: {data["b_sees_a_as"]})',
                  fontsize=14)
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(15, 40)

    # Bottom plot: Rolling win rate
    ax2.plot(games, [r * 100 for r in data['rolling_win_rate']],
             label=f'V3-A Rolling Win Rate ({ROLLING_WINDOW_SIZE} games)', color='green', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (fair)')

    final_wr = data['final_a_win_rate'] * 100
    ax2.axhline(y=final_wr, color='green', linestyle=':', alpha=0.7,
                label=f'Final: {final_wr:.1f}%')

    ax2.set_xlabel('Game Number', fontsize=12)
    ax2.set_ylabel('V3-A Win Rate (%)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(30, 70)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_learning_phase_grid(filename):
    """Generate 2x2 grid showing first 100 games against each opponent type."""
    print("\nTracking learning phase (100 games each)...")

    opponents = ["MANIAC", "AGGRESSIVE", "PASSIVE", "CALLING_STATION"]
    all_data = {}

    for opp in opponents:
        print(f"  vs {opp}...")
        all_data[opp] = track_v3_adaptation(opp, num_games=100)

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, opp in enumerate(opponents):
        ax = axes[idx]
        data = all_data[opp]
        games = data['games']

        # Plot thresholds
        ax.plot(games, [t * 100 for t in data['call_threshold']],
                label='Call Threshold', color='blue', linewidth=2)
        ax.plot(games, [t * 100 for t in data['raise_threshold']],
                label='Raise Threshold', color='red', linewidth=2)

        # GTO baselines
        ax.axhline(y=25, color='blue', linestyle='--', alpha=0.3)
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.3)

        ax.set_title(f'vs {opp}', fontsize=12, weight='bold')
        ax.set_xlabel('Game')
        ax.set_ylabel('Threshold (%)')
        ax.set_ylim(15, 40)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper right')

        # Add final values annotation
        final_call = data['call_threshold'][-1] * 100
        final_raise = data['raise_threshold'][-1] * 100
        ax.text(0.95, 0.05, f'Call: {final_call:.0f}%\nRaise: {final_raise:.0f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('V3 ADAPTIVE: Learning Phase (First 100 Games)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_adaptation(data, filename):
    """Generate adaptation chart with two subplots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    games = data['games']
    opponent = data['opponent']

    # Top plot: Threshold evolution
    ax1.plot(games, [t * 100 for t in data['call_threshold']],
             label='Call Threshold', color='blue', linewidth=2)
    ax1.plot(games, [t * 100 for t in data['raise_threshold']],
             label='Raise Threshold', color='red', linewidth=2)

    # GTO baselines
    ax1.axhline(y=25, color='blue', linestyle='--', alpha=0.5, label='GTO Call (25%)')
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='GTO Raise (30%)')

    ax1.set_ylabel('Threshold (%)', fontsize=12)
    ax1.set_title(f'ADAPTIVE Learning Against {opponent}', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(10, 50)

    # Bottom plot: Rolling win rate
    ax2.plot(games, [r * 100 for r in data['rolling_win_rate']],
             label=f'Rolling Win Rate ({ROLLING_WINDOW_SIZE} games)', color='green', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')

    final_wr = data['final_win_rate'] * 100
    ax2.axhline(y=final_wr, color='green', linestyle=':', alpha=0.7,
                label=f'Final: {final_wr:.1f}%')

    ax2.set_xlabel('Game Number', fontsize=12)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(30, 80)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    """Generate all visualizations."""

    # Create output directory
    os.makedirs('visuals', exist_ok=True)

    print("\n" + "=" * 60)
    print("LIAR'S DICE VISUALIZATION")
    print(f"(Running with {DEFAULT_GAMES_PER_MATCHUP} games per matchup)")
    print("=" * 60)

    # 1. Round Robin
    print("\n[1/3] Running Round Robin Tournament...")
    matrix, wins_matrix, games_matrix = run_round_robin()
    plot_heatmap(matrix, PERSONALITY_ORDER, 'visuals/round_robin_heatmap.png',
                 wins_matrix, games_matrix)

    # Print standings
    standings = get_round_robin_standings(matrix)
    print("\nROUND ROBIN STANDINGS:")
    print("-" * 40)
    for i, (personality, win_rate) in enumerate(standings, 1):
        print(f"  {i}. {personality}: {win_rate:.1%}")

    # 2. Tournament Bracket
    print("\n[2/3] Running Tournament Bracket...")
    seeding = [p for p, _ in standings]  # Seed by round robin performance
    bracket_results = run_bracket_tournament(seeding)
    plot_bracket(bracket_results, 'visuals/tournament_bracket.png')

    # 3. V3 Adaptation Charts
    print("\n[3/3] Tracking V3 Adaptation...")

    # Track against MANIAC (should show clear adaptation)
    maniac_data = track_v3_adaptation("MANIAC", num_games=500)
    plot_adaptation(maniac_data, 'visuals/v3_adaptation_vs_maniac.png')

    # Track against PASSIVE
    passive_data = track_v3_adaptation("PASSIVE", num_games=500)
    plot_adaptation(passive_data, 'visuals/v3_adaptation_vs_passive.png')

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - visuals/round_robin_heatmap.png")
    print("  - visuals/tournament_bracket.png")
    print("  - visuals/v3_adaptation_vs_maniac.png")
    print("  - visuals/v3_adaptation_vs_passive.png")


if __name__ == "__main__":
    main()
