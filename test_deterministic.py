#!/usr/bin/env python3
"""Fast tournament testing with deterministic agents (no LLM)."""

import sys
import time
from collections import defaultdict

from game.engine import GameEngine
from agents.deterministic_agent import DeterministicV2Agent, DeterministicV3Agent
from agents.base import Agent
from stats_utils import (
    format_win_rate_with_ci,
    is_significantly_better,
    is_significantly_different,
    margin_of_error
)


def play_game(engine: GameEngine, agents: list[Agent]) -> tuple[int, int]:
    """Play a single game. Returns (winner_index, rounds_played)."""
    engine.start_game()

    while not engine.is_game_over:
        current = engine.state.current_player
        view = engine.get_player_view(current)
        action = agents[current].get_action(view)
        result = engine.apply_action(current, action)

        if result:
            for agent in agents:
                agent.notify_round_result(result)

    return engine.state.winner, engine.state.round_num


def run_tournament(
    agent1: Agent,
    agent2: Agent,
    num_games: int,
    verbose: bool = False
) -> dict:
    """Run a tournament and return statistics."""
    agents = [agent1, agent2]
    wins = [0, 0]
    total_rounds = 0

    start_time = time.time()

    for game_num in range(1, num_games + 1):
        engine = GameEngine(starting_dice=5)
        engine.initialize_game(num_players=2)

        winner, rounds = play_game(engine, agents)
        wins[winner] += 1
        total_rounds += rounds

        if verbose and game_num % 100 == 0:
            elapsed = time.time() - start_time
            rate = game_num / elapsed
            print(f"Game {game_num}/{num_games}: "
                  f"{agent1.name}={wins[0]} {agent2.name}={wins[1]} "
                  f"({rate:.1f} games/sec)")

    elapsed = time.time() - start_time

    return {
        "agent1_wins": wins[0],
        "agent2_wins": wins[1],
        "total_games": num_games,
        "agent1_win_rate": wins[0] / num_games,
        "agent2_win_rate": wins[1] / num_games,
        "avg_rounds": total_rounds / num_games,
        "elapsed_seconds": elapsed,
        "games_per_second": num_games / elapsed,
    }


def test_v3_vs_v2():
    """Test V3 (with memory) vs V2 (stateless)."""
    print("\n" + "=" * 60)
    print("TEST: DeterministicV3 (memory) vs DeterministicV2 (stateless)")
    print("=" * 60)

    # V3 with memory enabled
    v3 = DeterministicV3Agent("V3-Memory", memory_enabled=True)
    v2 = DeterministicV2Agent("V2-Static")

    results = run_tournament(v3, v2, num_games=1000, verbose=True)

    print(f"\n--- Results ---")
    print(f"V3 wins: {results['agent1_wins']} ({results['agent1_win_rate']:.1%})")
    print(f"V2 wins: {results['agent2_wins']} ({results['agent2_win_rate']:.1%})")
    print(f"Avg rounds/game: {results['avg_rounds']:.1f}")
    print(f"Speed: {results['games_per_second']:.1f} games/sec")

    # Show V3's learned model
    print(f"\n--- V3's Opponent Model ---")
    summary = v3.get_model_summary(1)  # opponent_id=1 for V2
    if summary:
        print(summary)

    return results


def test_v3_vs_v3():
    """Test V3 vs V3 (mutual adaptation)."""
    print("\n" + "=" * 60)
    print("TEST: DeterministicV3 vs DeterministicV3 (mutual adaptation)")
    print("=" * 60)

    v3a = DeterministicV3Agent("V3-A", memory_enabled=True)
    v3b = DeterministicV3Agent("V3-B", memory_enabled=True)

    results = run_tournament(v3a, v3b, num_games=1000, verbose=True)

    print(f"\n--- Results ---")
    print(f"V3-A wins: {results['agent1_wins']} ({results['agent1_win_rate']:.1%})")
    print(f"V3-B wins: {results['agent2_wins']} ({results['agent2_win_rate']:.1%})")

    print(f"\n--- V3-A's Model of V3-B ---")
    print(v3a.get_model_summary(1) or "No data")

    print(f"\n--- V3-B's Model of V3-A ---")
    print(v3b.get_model_summary(0) or "No data")

    return results


def test_v2_vs_v2():
    """Baseline: V2 vs V2 (should be ~50/50)."""
    print("\n" + "=" * 60)
    print("TEST: DeterministicV2 vs DeterministicV2 (baseline)")
    print("=" * 60)

    v2a = DeterministicV2Agent("V2-A")
    v2b = DeterministicV2Agent("V2-B")

    results = run_tournament(v2a, v2b, num_games=1000, verbose=True)

    print(f"\n--- Results ---")
    print(f"V2-A wins: {results['agent1_wins']} ({results['agent1_win_rate']:.1%})")
    print(f"V2-B wins: {results['agent2_wins']} ({results['agent2_win_rate']:.1%})")
    print(f"Expected: ~50% each (first-mover advantage possible)")

    return results


def test_exploitation_over_time():
    """Show how V3's advantage grows with more observations."""
    print("\n" + "=" * 60)
    print("TEST: V3 exploitation over time (10 separate 100-game tournaments)")
    print("=" * 60)

    for tournament in range(1, 11):
        # Fresh agents each tournament
        v3 = DeterministicV3Agent(f"V3-T{tournament}", memory_enabled=True)
        v2 = DeterministicV2Agent(f"V2-T{tournament}")

        results = run_tournament(v3, v2, num_games=100, verbose=False)

        model = v3._opponent_models.get(1)
        obs = model.observation_count if model else 0
        opp_type = model.get_opponent_type().value if model else "N/A"

        print(f"Tournament {tournament:2d}: "
              f"V3={results['agent1_wins']:3d} V2={results['agent2_wins']:3d} "
              f"(V3: {results['agent1_win_rate']:.0%}) "
              f"| obs={obs:3d} type={opp_type}")


def test_v3_vs_2x2_quadrants():
    """Test V3 against all 4 opponent quadrants + balanced.

    2x2 Matrix:
                          CALLING BEHAVIOR
                          Low (<35%)        High (>65%)
                         ┌─────────────────┬─────────────────┐
    BIDDING     High     │    MANIAC       │   AGGRESSIVE    │
    BEHAVIOR    (>65%    │ bluffs, passive │ bluffs, calls   │
    (bluff      bluff)   │                 │                 │
    rate)                ├─────────────────┼─────────────────┤
                Low      │    PASSIVE      │ CALLING STATION │
                (<35%    │ honest, passive │ honest, calls   │
                bluff)   │                 │                 │
                         └─────────────────┴─────────────────┘

    V3 exploitation strategy per quadrant:
    - MANIAC: call more, bluff more
    - AGGRESSIVE: call more, don't bluff
    - PASSIVE: call less, bluff more
    - CALLING_STATION: call less, don't bluff
    - BALANCED: stay GTO
    """
    print("\n" + "=" * 60)
    print("TEST: V3 vs 2x2 OPPONENT QUADRANTS")
    print("=" * 60)

    # Define opponents for each quadrant
    # High bluff = low safe_raise_threshold (bluffs with low P)
    # High call = high call_threshold (calls more often)
    opponents = {
        "MANIAC": DeterministicV2Agent(
            "MANIAC",
            call_threshold=0.10,      # Low call → passive
            safe_raise_threshold=0.10  # Bluffs freely
        ),
        "AGGRESSIVE": DeterministicV2Agent(
            "AGGRESSIVE",
            call_threshold=0.50,       # High call
            safe_raise_threshold=0.10  # Bluffs freely
        ),
        "PASSIVE": DeterministicV2Agent(
            "PASSIVE",
            call_threshold=0.10,       # Low call → passive
            safe_raise_threshold=0.50  # Only raises with good hands
        ),
        "CALLING_STATION": DeterministicV2Agent(
            "CALLING_STATION",
            call_threshold=0.50,       # High call
            safe_raise_threshold=0.50  # Only raises with good hands
        ),
        "BALANCED": DeterministicV2Agent(
            "BALANCED",
            call_threshold=0.25,       # GTO default
            safe_raise_threshold=0.30  # GTO default
        ),
    }

    results = {}
    v2_results = {}

    # Test V3 against each opponent type
    for opp_type, opponent in opponents.items():
        print(f"\n--- V3 vs {opp_type} ---")

        v3 = DeterministicV3Agent("V3", memory_enabled=True)
        res = run_tournament(v3, opponent, num_games=500, verbose=False)
        results[opp_type] = res

        # Get what V3 learned
        model = v3._opponent_models.get(1)
        if model:
            detected_type = model.get_opponent_type().value
            call_th, raise_th = model.get_adjusted_thresholds()
            print(f"  V3 detected: {detected_type}")
            print(f"  V3 rates:   bluff={model.bluff_rate:.0%}, call={model.call_rate:.0%}")
            print(f"  Adj thresholds: call={call_th:.0%}, raise={raise_th:.0%}")
        print(f"  Result: V3={res['agent1_win_rate']:.1%} vs {opp_type}={res['agent2_win_rate']:.1%}")

        # Run V2 baseline for comparison
        v2_opp = DeterministicV2Agent(
            opponent.name,
            call_threshold=opponent.call_threshold,
            safe_raise_threshold=opponent.safe_raise_threshold
        )
        v2 = DeterministicV2Agent("V2")
        v2_res = run_tournament(v2, v2_opp, num_games=500, verbose=False)
        v2_results[opp_type] = v2_res

    # Summary comparison with confidence intervals
    print("\n" + "=" * 60)
    print("SUMMARY: V3 vs V2 BASELINE (with 95% CI)")
    print("=" * 60)
    n_games = 500  # From run_tournament calls above
    print(f"{'Opponent':<16} {'V3 Win Rate':>16} {'V2 Win Rate':>16} {'Diff':>10} {'Sig?':>6}")
    print("-" * 64)

    for opp_type in opponents:
        v3_wins = results[opp_type]['agent1_wins']
        v2_wins = v2_results[opp_type]['agent1_wins']
        v3_wr = results[opp_type]['agent1_win_rate']
        v2_wr = v2_results[opp_type]['agent1_win_rate']
        advantage = v3_wr - v2_wr

        v3_ci = format_win_rate_with_ci(v3_wins, n_games)
        v2_ci = format_win_rate_with_ci(v2_wins, n_games)

        # Statistical significance test
        sig = is_significantly_different(v3_wins, n_games, v2_wins, n_games)
        sig_marker = "*" if sig else ""

        print(f"{opp_type:<16} {v3_ci:>16} {v2_ci:>16} {advantage:>+9.1%} {sig_marker:>6}")

    # Check success criteria with statistical significance
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK (with significance testing)")
    print("=" * 60)
    print("(* = statistically significant at p < 0.05)")

    success = True
    for opp_type in ["MANIAC", "AGGRESSIVE", "PASSIVE", "CALLING_STATION"]:
        v3_wins = results[opp_type]['agent1_wins']
        v2_wins = v2_results[opp_type]['agent1_wins']
        v3_wr = results[opp_type]['agent1_win_rate']
        v2_wr = v2_results[opp_type]['agent1_win_rate']
        improvement = v3_wr - v2_wr

        # Use significance test instead of arbitrary threshold
        sig_better = is_significantly_better(v3_wins, n_games, v2_wins, n_games)
        status = "✓" if sig_better else "✗"
        sig_marker = "*" if sig_better else ""
        print(f"{status} {opp_type}: V3 beats V2 baseline → {improvement:+.1%} {sig_marker}")
        if not sig_better:
            success = False

    balanced_wins = results["BALANCED"]['agent1_wins']
    balanced_wr = results["BALANCED"]['agent1_win_rate']
    # Not significantly different from 50% = success
    sig_diff = is_significantly_different(balanced_wins, n_games, n_games // 2, n_games)
    balanced_status = "✓" if not sig_diff else "✗"
    print(f"{balanced_status} BALANCED: V3 ties (~50%) → {balanced_wr:.1%} {'(sig diff from 50%)' if sig_diff else ''}")
    if sig_diff:
        success = False

    print("\n" + ("ALL CRITERIA MET ✓" if success else "SOME CRITERIA FAILED ✗"))

    return results, v2_results


def test_v3_vs_weak_opponents():
    """Legacy test - redirects to 2x2 quadrant test."""
    return test_v3_vs_2x2_quadrants()


def test_cliff_vs_gradual():
    """Compare cliff threshold vs gradual ramp-up approaches.

    Tests both continuous adjustment modes against all opponent types:
    - Cliff: No adjustment until 20 observations, then full strength
    - Gradual: Scale adjustments from 0% to 100% as observations grow
    """
    print("\n" + "=" * 60)
    print("TEST: CLIFF vs GRADUAL RAMP-UP COMPARISON")
    print("=" * 60)

    # Define opponents for each quadrant
    opponents = {
        "MANIAC": DeterministicV2Agent(
            "MANIAC",
            call_threshold=0.10,
            safe_raise_threshold=0.10
        ),
        "AGGRESSIVE": DeterministicV2Agent(
            "AGGRESSIVE",
            call_threshold=0.50,
            safe_raise_threshold=0.10
        ),
        "PASSIVE": DeterministicV2Agent(
            "PASSIVE",
            call_threshold=0.10,
            safe_raise_threshold=0.50
        ),
        "CALLING_STATION": DeterministicV2Agent(
            "CALLING_STATION",
            call_threshold=0.50,
            safe_raise_threshold=0.50
        ),
        "BALANCED": DeterministicV2Agent(
            "BALANCED",
            call_threshold=0.25,
            safe_raise_threshold=0.30
        ),
    }

    cliff_results = {}
    gradual_results = {}
    v2_results = {}

    for opp_type, opponent in opponents.items():
        print(f"\n--- Testing against {opp_type} ---")

        # V3 with cliff threshold (old behavior)
        v3_cliff = DeterministicV3Agent("V3-Cliff", use_gradual_rampup=False)
        # Create fresh opponent for fair comparison
        opp_cliff = DeterministicV2Agent(
            opponent.name,
            call_threshold=opponent.call_threshold,
            safe_raise_threshold=opponent.safe_raise_threshold
        )
        cliff_res = run_tournament(v3_cliff, opp_cliff, num_games=500, verbose=False)
        cliff_results[opp_type] = cliff_res

        # V3 with gradual ramp-up (new behavior)
        v3_gradual = DeterministicV3Agent("V3-Gradual", use_gradual_rampup=True)
        opp_gradual = DeterministicV2Agent(
            opponent.name,
            call_threshold=opponent.call_threshold,
            safe_raise_threshold=opponent.safe_raise_threshold
        )
        gradual_res = run_tournament(v3_gradual, opp_gradual, num_games=500, verbose=False)
        gradual_results[opp_type] = gradual_res

        # V2 baseline for comparison
        v2 = DeterministicV2Agent("V2")
        opp_v2 = DeterministicV2Agent(
            opponent.name,
            call_threshold=opponent.call_threshold,
            safe_raise_threshold=opponent.safe_raise_threshold
        )
        v2_res = run_tournament(v2, opp_v2, num_games=500, verbose=False)
        v2_results[opp_type] = v2_res

        print(f"  V2 (GTO):     {v2_res['agent1_win_rate']:.1%}")
        print(f"  V3-Cliff:     {cliff_res['agent1_win_rate']:.1%} (diff: {cliff_res['agent1_win_rate'] - v2_res['agent1_win_rate']:+.1%})")
        print(f"  V3-Gradual:   {gradual_res['agent1_win_rate']:.1%} (diff: {gradual_res['agent1_win_rate'] - v2_res['agent1_win_rate']:+.1%})")

    # Summary comparison with CIs and significance
    print("\n" + "=" * 60)
    print("SUMMARY: CLIFF vs GRADUAL (with 95% CI)")
    print("=" * 60)
    n_games = 500  # From run_tournament calls above
    print(f"{'Opponent':<14} {'V2 (GTO)':>14} {'V3-Cliff':>14} {'V3-Gradual':>14} {'Winner':>10}")
    print("-" * 66)

    cliff_wins = 0
    gradual_wins = 0

    for opp_type in opponents:
        v2_w = v2_results[opp_type]['agent1_wins']
        cliff_w = cliff_results[opp_type]['agent1_wins']
        gradual_w = gradual_results[opp_type]['agent1_wins']

        v2_ci = format_win_rate_with_ci(v2_w, n_games)
        cliff_ci = format_win_rate_with_ci(cliff_w, n_games)
        gradual_ci = format_win_rate_with_ci(gradual_w, n_games)

        # Use significance test for winner determination
        cliff_better = is_significantly_better(cliff_w, n_games, gradual_w, n_games)
        gradual_better = is_significantly_better(gradual_w, n_games, cliff_w, n_games)

        if cliff_better:
            winner = "Cliff*"
            cliff_wins += 1
        elif gradual_better:
            winner = "Gradual*"
            gradual_wins += 1
        else:
            winner = "Tie"

        print(f"{opp_type:<14} {v2_ci:>14} {cliff_ci:>14} {gradual_ci:>14} {winner:>10}")

    print("\n" + "-" * 66)
    print(f"Score: Cliff={cliff_wins} Gradual={gradual_wins} Ties={5 - cliff_wins - gradual_wins}")
    print("(* = statistically significant at p < 0.05)")

    # Check if both beat GTO baseline with significance
    print("\n" + "=" * 60)
    print("BEATS V2 GTO BASELINE? (with significance)")
    print("=" * 60)

    for opp_type in ["MANIAC", "AGGRESSIVE", "PASSIVE", "CALLING_STATION"]:
        v2_w = v2_results[opp_type]['agent1_wins']
        cliff_w = cliff_results[opp_type]['agent1_wins']
        gradual_w = gradual_results[opp_type]['agent1_wins']

        cliff_diff = cliff_results[opp_type]['agent1_win_rate'] - v2_results[opp_type]['agent1_win_rate']
        gradual_diff = gradual_results[opp_type]['agent1_win_rate'] - v2_results[opp_type]['agent1_win_rate']

        cliff_sig = is_significantly_better(cliff_w, n_games, v2_w, n_games)
        gradual_sig = is_significantly_better(gradual_w, n_games, v2_w, n_games)

        cliff_status = "✓*" if cliff_sig else ("✓" if cliff_diff > 0 else "✗")
        gradual_status = "✓*" if gradual_sig else ("✓" if gradual_diff > 0 else "✗")

        print(f"{opp_type}: Cliff {cliff_status} ({cliff_diff:+.1%}) | Gradual {gradual_status} ({gradual_diff:+.1%})")

    print("(* = statistically significant at p < 0.05)")

    return cliff_results, gradual_results, v2_results


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("#  DETERMINISTIC AGENT TESTING")
    print("#  (No LLM calls - instant results)")
    print("#" * 60)

    # Baseline test
    baseline = test_v2_vs_v2()

    # Main test: V3 vs V2
    v3_vs_v2 = test_v3_vs_v2()

    # V3 vs V3
    v3_vs_v3 = test_v3_vs_v3()

    # V3 vs WEAK opponents (where exploitation matters!)
    test_v3_vs_weak_opponents()

    # NEW: Cliff vs Gradual comparison
    test_cliff_vs_gradual()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"V2 vs V2 (baseline): {baseline['agent1_win_rate']:.1%} vs {baseline['agent2_win_rate']:.1%}")
    print(f"V3 vs V2 (memory):   {v3_vs_v2['agent1_win_rate']:.1%} vs {v3_vs_v2['agent2_win_rate']:.1%}")
    print(f"V3 vs V3 (mutual):   {v3_vs_v3['agent1_win_rate']:.1%} vs {v3_vs_v3['agent2_win_rate']:.1%}")

    # Calculate improvement
    improvement = v3_vs_v2['agent1_win_rate'] - 0.5
    print(f"\nV3 improvement over V2 (optimal opponent): {improvement:+.1%}")
    print("NOTE: V3 shines against WEAK opponents, not optimal ones!")


if __name__ == "__main__":
    main()
