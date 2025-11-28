#!/usr/bin/env python3
"""
Interactive CLI for Liar's Dice.

Game Modes:
    1. Play - Human vs AI
    2. Watch - AI vs AI (verbose)
    3. Tournament - Fast head-to-head

Personality Types:
    MANIAC: Bluffs freely, never calls
    AGGRESSIVE: Bluffs freely, calls everything
    PASSIVE: Plays honest, never calls
    CALLING_STATION: Plays honest, calls everything
    OPTIMAL: Tournament-winning thresholds (default)
"""

import time
from game.engine import GameEngine
from agents.base import Agent
from agents.human import HumanAgent
from agents.deterministic_agent import (
    DeterministicV2Agent,
    create_personality_agent,
    PERSONALITIES
)


def get_choice(prompt: str, options: list[str]) -> int:
    """Get a numbered choice from the user."""
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice = input("\n> ").strip()
            num = int(choice)
            if 1 <= num <= len(options):
                return num
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a number")


def get_number(prompt: str, default: int) -> int:
    """Get a number from the user with a default."""
    while True:
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            if not value:
                return default
            return int(value)
        except ValueError:
            print("Please enter a number")


def get_personality_options() -> list[str]:
    """Return list of personality options for display."""
    descriptions = {
        "MANIAC": "bluffs freely, never calls",
        "AGGRESSIVE": "bluffs freely, calls everything",
        "PASSIVE": "plays honest, never calls",
        "CALLING_STATION": "plays honest, calls everything",
        "OPTIMAL": "tournament winner (default)",
    }
    return [f"{p} ({descriptions[p]})" for p in PERSONALITIES.keys()]


def get_personality_from_choice(choice: int) -> str:
    """Convert menu choice to personality name."""
    return list(PERSONALITIES.keys())[choice - 1]


def play_game(
    engine: GameEngine,
    agents: list[Agent],
    fast_mode: bool = False,
    game_label: str = ""
) -> tuple[int, int]:
    """Play a single game to completion.

    Returns:
        (winner_index, rounds_played)
    """
    engine.start_game()
    last_round = 0
    progress_parts = []

    while not engine.is_game_over:
        current = engine.state.current_player
        view = engine.get_player_view(current)

        # Show round header when round changes (verbose mode only)
        if not fast_mode and engine.state.round_num != last_round:
            last_round = engine.state.round_num
            print(f"\n{'=' * 50}")
            print(f"              === ROUND {last_round} ===")
            dice_status = " | ".join(
                f"{agents[i].name}: {p.dice_count} dice"
                for i, p in enumerate(engine.state.players)
            )
            print(f"  {dice_status}")
            print("=" * 50)

        if not fast_mode:
            print(f"\n>>> {agents[current].name}'s turn")

        action = agents[current].get_action(view)

        if not fast_mode:
            if action.bid:
                print(f">>> {agents[current].name} bids: {action.bid}")
            else:
                print(f">>> {agents[current].name} CALLS!")

        result = engine.apply_action(current, action)
        last_round = engine.state.round_num

        if result:
            for agent in agents:
                agent.notify_round_result(result)

            if fast_mode:
                d1 = engine.state.players[0].dice_count
                d2 = engine.state.players[1].dice_count
                progress_parts.append(f"{d1}v{d2}")
                progress_str = " -> ".join(progress_parts[-6:])
                print(f"\r{game_label} {progress_str}...", end="", flush=True)
            else:
                print("\n" + "-" * 50)
                print("ROUND RESULT")
                print("-" * 50)

                for player_id, dice in result.all_dice.items():
                    print(f"  {agents[player_id].name}'s dice: {dice}")

                print(f"\n  Challenged bid: {result.challenged_bid}")
                print(f"  Actual count of {result.challenged_bid.face_value}s: {result.actual_count}")

                if result.bid_was_valid:
                    print(f"\n  Bid was VALID! {agents[result.caller_id].name} loses a die.")
                else:
                    print(f"\n  Bid was a BLUFF! {agents[result.bidder_id].name} loses a die.")

                print("-" * 50)

        if not fast_mode:
            time.sleep(0.3)

    return engine.state.winner, last_round


def play_game_silent(engine: GameEngine, agents: list[Agent]) -> tuple[int, int]:
    """Play a game with no output (for fast tournaments).

    Returns:
        (winner_index, rounds_played)
    """
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


def mode_play():
    """Human vs AI mode."""
    print("\n" + "=" * 50)
    print("         PLAY: Human vs AI")
    print("=" * 50)

    options = get_personality_options()
    choice = get_choice("\nSelect AI personality:", options)
    personality = get_personality_from_choice(choice)

    agents = [
        HumanAgent("You"),
        create_personality_agent(personality, quiet=False)
    ]

    print(f"\nPlaying against: {agents[1].name}")

    engine = GameEngine(starting_dice=5)
    engine.initialize_game(num_players=2)

    try:
        winner, rounds = play_game(engine, agents, fast_mode=False)

        print(f"\n{'=' * 50}")
        print("           === GAME OVER ===")
        if winner == 0:
            print("  YOU WIN!")
        else:
            print(f"  {agents[1].name} wins!")
        print(f"  Rounds played: {rounds}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nGame cancelled.")


def mode_watch():
    """AI vs AI mode (verbose)."""
    print("\n" + "=" * 50)
    print("         WATCH: AI vs AI")
    print("=" * 50)

    options = get_personality_options()
    p1_choice = get_choice("\nAgent 1 personality:", options)
    p2_choice = get_choice("Agent 2 personality:", options)

    p1_name = get_personality_from_choice(p1_choice)
    p2_name = get_personality_from_choice(p2_choice)

    agents = [
        create_personality_agent(p1_name, quiet=False),
        create_personality_agent(p2_name, quiet=False)
    ]

    print(f"\n{agents[0].name} vs {agents[1].name}")
    print("Press Ctrl+C to stop\n")

    engine = GameEngine(starting_dice=5)
    engine.initialize_game(num_players=2)

    try:
        winner, rounds = play_game(engine, agents, fast_mode=False)

        print(f"\n{'=' * 50}")
        print("           === GAME OVER ===")
        print(f"  Winner: {agents[winner].name}")
        print(f"  Rounds played: {rounds}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nMatch cancelled.")


def mode_tournament():
    """Fast tournament mode."""
    print("\n" + "=" * 50)
    print("         TOURNAMENT: Fast Mode")
    print("=" * 50)

    options = get_personality_options()
    p1_choice = get_choice("\nAgent 1 personality:", options)
    p2_choice = get_choice("Agent 2 personality:", options)

    p1_name = get_personality_from_choice(p1_choice)
    p2_name = get_personality_from_choice(p2_choice)

    num_games = get_number("Number of games", 100)

    agents = [
        create_personality_agent(p1_name, quiet=True),
        create_personality_agent(p2_name, quiet=True)
    ]

    print(f"\n{agents[0].name} vs {agents[1].name}")
    print(f"Running {num_games} games...\n")

    wins = [0, 0]
    total_rounds = 0

    try:
        for game_num in range(1, num_games + 1):
            engine = GameEngine(starting_dice=5)
            engine.initialize_game(num_players=2)

            winner, rounds = play_game_silent(engine, agents)
            wins[winner] += 1
            total_rounds += rounds

            # Show progress bar every 1% or at least every 10 games
            update_interval = max(1, num_games // 100)
            if game_num % update_interval == 0 or game_num == num_games:
                pct = game_num / num_games
                bar_len = 40
                filled = int(bar_len * pct)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r[{bar}] {game_num}/{num_games} ({pct*100:.0f}%)", end="", flush=True)

        print()  # Newline after progress bar

        # Results summary
        print("\n" + "=" * 50)
        print("           TOURNAMENT RESULTS")
        print("=" * 50)
        print(f"  {agents[0].name}: {wins[0]} wins ({wins[0]/num_games*100:.1f}%)")
        print(f"  {agents[1].name}: {wins[1]} wins ({wins[1]/num_games*100:.1f}%)")
        print(f"  Avg rounds/game: {total_rounds/num_games:.1f}")

        if wins[0] > wins[1]:
            print(f"\n  Winner: {agents[0].name}")
        elif wins[1] > wins[0]:
            print(f"\n  Winner: {agents[1].name}")
        else:
            print("\n  Result: TIE")
        print("=" * 50)

    except KeyboardInterrupt:
        print(f"\n\nTournament stopped after {game_num-1} games.")
        if game_num > 1:
            print(f"Partial results: A1={wins[0]}, A2={wins[1]}")


def main():
    print("\n" + "=" * 50)
    print("         LIAR'S DICE")
    print("=" * 50)
    print("\nPersonality Types:")
    print("  MANIAC - bluffs freely, never calls")
    print("  AGGRESSIVE - bluffs freely, calls everything")
    print("  PASSIVE - plays honest, never calls")
    print("  CALLING_STATION - plays honest, calls everything")
    print("  OPTIMAL - tournament winner (56.5% win rate)")

    mode = get_choice(
        "\nSelect game mode:",
        [
            "Play - Human vs AI",
            "Watch - AI vs AI (verbose)",
            "Tournament - Fast head-to-head"
        ]
    )

    if mode == 1:
        mode_play()
    elif mode == 2:
        mode_watch()
    elif mode == 3:
        mode_tournament()


if __name__ == "__main__":
    main()
