#!/usr/bin/env python3
"""Fast tournament testing with deterministic agents."""

import sys
import time
from collections import defaultdict

from game.engine import GameEngine
from agents.deterministic_agent import DeterministicV2Agent, create_personality_agent
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


def test_baseline():
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


def test_optimal_vs_personalities():
    """Test OPTIMAL against all other personality types."""
    print("\n" + "=" * 60)
    print("TEST: OPTIMAL vs ALL PERSONALITIES")
    print("=" * 60)

    personalities = ["MANIAC", "AGGRESSIVE", "PASSIVE", "CALLING_STATION"]
    results = {}

    for personality in personalities:
        print(f"\n--- OPTIMAL vs {personality} ---")

        optimal = create_personality_agent("OPTIMAL")
        opponent = create_personality_agent(personality)

        res = run_tournament(optimal, opponent, num_games=500, verbose=False)
        results[personality] = res

        print(f"  OPTIMAL: {res['agent1_win_rate']:.1%}")
        print(f"  {personality}: {res['agent2_win_rate']:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: OPTIMAL vs PERSONALITIES")
    print("=" * 60)
    print(f"{'Opponent':<18} {'OPTIMAL Win Rate':>16} {'Opponent Win Rate':>18}")
    print("-" * 54)

    for personality in personalities:
        res = results[personality]
        print(f"{personality:<18} {res['agent1_win_rate']:>16.1%} {res['agent2_win_rate']:>18.1%}")

    return results


def test_personality_matchups():
    """Test all personality matchups to verify they work."""
    print("\n" + "=" * 60)
    print("TEST: ALL PERSONALITY MATCHUPS (100 games each)")
    print("=" * 60)

    personalities = ["MANIAC", "AGGRESSIVE", "PASSIVE", "CALLING_STATION", "OPTIMAL"]

    print(f"\n{'':>16}", end="")
    for p in personalities[:4]:
        print(f"{p[:8]:>10}", end="")
    print()
    print("-" * 56)

    for p1 in personalities:
        print(f"{p1:<16}", end="")
        for p2 in personalities[:4]:
            if p1 == p2:
                print(f"{'---':>10}", end="")
            else:
                a1 = create_personality_agent(p1)
                a2 = create_personality_agent(p2)
                res = run_tournament(a1, a2, num_games=100, verbose=False)
                print(f"{res['agent1_win_rate']:>10.0%}", end="")
        print()

    print("\n(Rows = Agent 1, Columns = Agent 2, showing Agent 1 win rate)")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("#  DETERMINISTIC AGENT TESTING")
    print("#  (No LLM calls - instant results)")
    print("#" * 60)

    # Baseline test
    baseline = test_baseline()

    # OPTIMAL vs all personalities
    test_optimal_vs_personalities()

    # All matchups
    test_personality_matchups()

    # Summary
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
