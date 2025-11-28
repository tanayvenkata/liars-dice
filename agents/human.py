"""Human player agent for terminal input."""

from game.types import PlayerView, Action, ActionType, Bid, RoundResult
from agents.base import Agent


class HumanAgent(Agent):
    """Human player via terminal input."""

    def get_action(self, view: PlayerView) -> Action:
        """Get action from human via terminal."""
        self._display_state(view)

        while True:
            try:
                user_input = input("\nYour action (e.g., 'bid 3 4' or 'call'): ").strip().lower()
                action = self._parse_input(user_input, view)
                return action
            except ValueError as e:
                print(f"Invalid input: {e}")
                print("Try again. Format: 'bid <quantity> <face>' or 'call'")

    def _display_state(self, view: PlayerView) -> None:
        """Display the current game state to the human player."""
        print("\n" + "=" * 50)
        print(f"Round {view.round_num}")
        print(f"Your dice: {sorted(view.own_dice)}")
        print(f"Opponent has {view.opponent_dice_count} dice")
        print(f"Total dice in play: {view.total_dice}")

        if view.current_bid:
            print(f"\nCurrent bid to beat: {view.current_bid}")
            print("You can BID higher or CALL")
        else:
            print("\nYou open the bidding")

    def _parse_input(self, user_input: str, view: PlayerView) -> Action:
        """Parse human input into an Action."""
        parts = user_input.split()

        if not parts:
            raise ValueError("Empty input")

        if parts[0] == "call":
            if view.current_bid is None:
                raise ValueError("Cannot call on opening bid")
            return Action(ActionType.CALL)

        if parts[0] == "bid":
            if len(parts) != 3:
                raise ValueError("Bid format: 'bid <quantity> <face>'")

            try:
                quantity = int(parts[1])
                face = int(parts[2])
            except ValueError:
                raise ValueError("Quantity and face must be numbers")

            bid = Bid(quantity=quantity, face_value=face)

            if not bid.is_valid_raise_over(view.current_bid):
                raise ValueError(f"Bid must be higher than {view.current_bid}")

            return Action(ActionType.BID, bid=bid)

        raise ValueError(f"Unknown command: {parts[0]}")

    def notify_round_result(self, result: RoundResult) -> None:
        """Display round result to human."""
        print("\n" + "-" * 50)
        print("ROUND RESULT")
        print(f"All dice revealed: {result.all_dice}")
        print(f"Challenged bid: {result.challenged_bid}")
        print(f"Actual count of {result.challenged_bid.face_value}s: {result.actual_count}")

        if result.bid_was_valid:
            print(f"The bid was VALID! Player {result.caller_id} loses a die.")
        else:
            print(f"The bid was a BLUFF! Player {result.bidder_id} loses a die.")
