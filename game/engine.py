"""Game engine for Liar's Dice - manages state and enforces rules."""

from typing import Optional

from game.types import (
    GameState, PlayerState, PlayerView, Action, ActionType,
    Bid, GamePhase, GameEvent, RoundResult
)


class GameEngine:
    """
    Manages game state and enforces rules for Liar's Dice.

    The engine is completely agnostic to how players make decisions.
    It only validates actions and updates state.
    """

    def __init__(self, starting_dice: int = 5):
        self.starting_dice = starting_dice
        self.state = GameState()
        self._current_round_bids: list[tuple[int, Bid]] = []  # Track bids for opponent modeling

    def initialize_game(self, num_players: int = 2) -> None:
        """Set up a new game with the specified number of players."""
        self.state = GameState(
            players=[
                PlayerState(player_id=i, dice_count=self.starting_dice)
                for i in range(num_players)
            ],
            phase=GamePhase.WAITING_TO_START
        )

    def start_game(self) -> None:
        """Start the game by rolling dice and beginning round 1."""
        if self.state.phase != GamePhase.WAITING_TO_START:
            raise ValueError("Game already started")

        self._start_new_round()

    def _start_new_round(self) -> None:
        """Begin a new round - roll dice and reset bid."""
        for player in self.state.players:
            player.roll_dice()

        self.state.current_bid = None
        self.state.phase = GamePhase.ROUND_IN_PROGRESS
        self._current_round_bids.clear()  # Reset bid history for new round

        # Log round start event
        self._add_event("round_start", None,
                        f"Round {self.state.round_num} begins. "
                        f"Total dice: {self.state.total_dice}")

    def get_player_view(self, player_id: int) -> PlayerView:
        """Get the game state from a specific player's perspective."""
        player = self.state.players[player_id]
        opponent = self.state.players[1 - player_id]  # Works for 2 players

        return PlayerView(
            player_id=player_id,
            own_dice=player.dice.copy(),
            opponent_dice_count=opponent.dice_count,
            current_bid=self.state.current_bid,
            round_num=self.state.round_num,
            total_dice=self.state.total_dice,
            history=self.state.history.copy()
        )

    def get_valid_actions(self, player_id: int) -> list[ActionType]:
        """Return list of valid action types for the given player."""
        if self.state.current_player != player_id:
            return []
        if self.state.phase != GamePhase.ROUND_IN_PROGRESS:
            return []

        actions = [ActionType.BID]
        if self.state.current_bid is not None:
            actions.append(ActionType.CALL)
        return actions

    def validate_action(self, player_id: int, action: Action) -> tuple[bool, str]:
        """
        Validate an action before applying it.

        Returns:
            (is_valid, error_message)
        """
        # Check it's this player's turn
        if self.state.current_player != player_id:
            return False, f"Not player {player_id}'s turn"

        # Check game is in progress
        if self.state.phase != GamePhase.ROUND_IN_PROGRESS:
            return False, "Round not in progress"

        # Validate specific action type
        if action.action_type == ActionType.CALL:
            if self.state.current_bid is None:
                return False, "Cannot call on first turn of round"
            return True, ""

        elif action.action_type == ActionType.BID:
            if action.bid is None:
                return False, "BID action requires a bid"

            # Check bid is valid raise
            if not action.bid.is_valid_raise_over(self.state.current_bid):
                return False, f"Bid {action.bid} does not raise over {self.state.current_bid}"

            # Check bid is not impossibly high
            if action.bid.quantity > self.state.total_dice:
                return False, f"Bid quantity {action.bid.quantity} exceeds total dice {self.state.total_dice}"

            return True, ""

        return False, f"Unknown action type: {action.action_type}"

    def apply_action(self, player_id: int, action: Action) -> Optional[RoundResult]:
        """
        Apply a validated action to the game state.

        Returns:
            RoundResult if the action was a CALL, None otherwise
        """
        is_valid, error = self.validate_action(player_id, action)
        if not is_valid:
            raise ValueError(f"Invalid action: {error}")

        if action.action_type == ActionType.BID:
            return self._apply_bid(player_id, action)
        else:
            return self._apply_call(player_id, action)

    def _apply_bid(self, player_id: int, action: Action) -> None:
        """Apply a bid action."""
        self.state.current_bid = action.bid
        self._current_round_bids.append((player_id, action.bid))  # Track for opponent modeling
        self._add_event("bid", player_id,
                        f"Player {player_id} bids {action.bid}")

        # Switch to other player
        self.state.current_player = 1 - player_id
        return None

    def _apply_call(self, player_id: int, action: Action) -> RoundResult:
        """Apply a call action and resolve the round."""
        self.state.phase = GamePhase.ROUND_ENDED

        bidder_id = 1 - player_id
        challenged_bid = self.state.current_bid

        # Count actual dice matching the bid's face value
        all_dice = {p.player_id: p.dice.copy() for p in self.state.players}
        actual_count = sum(
            1 for p in self.state.players
            for die in p.dice
            if die == challenged_bid.face_value
        )

        # Bid is valid if actual count >= bid quantity
        bid_was_valid = actual_count >= challenged_bid.quantity
        loser_id = player_id if bid_was_valid else bidder_id

        result = RoundResult(
            caller_id=player_id,
            bidder_id=bidder_id,
            challenged_bid=challenged_bid,
            actual_count=actual_count,
            all_dice=all_dice,
            loser_id=loser_id,
            bid_was_valid=bid_was_valid,
            bid_history=self._current_round_bids.copy()  # Include full bid history
        )

        # Log events
        self._add_event("call", player_id,
                        f"Player {player_id} calls! Challenging bid: {challenged_bid}")
        self._add_event("reveal", None,
                        f"Dice revealed: {all_dice}. "
                        f"Actual {challenged_bid.face_value}s: {actual_count}")
        self._add_event("round_result", loser_id,
                        f"Player {loser_id} loses a die. "
                        f"Bid was {'valid' if bid_was_valid else 'a bluff'}.")

        # Apply penalty
        eliminated = self.state.players[loser_id].lose_die()

        if eliminated:
            self.state.phase = GamePhase.GAME_OVER
            self.state.winner = 1 - loser_id
            self._add_event("game_over", self.state.winner,
                            f"Player {self.state.winner} wins!")
        else:
            # Prepare for next round
            self.state.round_num += 1
            self.state.current_player = loser_id  # Loser starts next round
            self._start_new_round()

        return result

    def _add_event(self, event_type: str, player_id: Optional[int], details: str) -> None:
        """Add an event to the game history."""
        event = GameEvent(
            round_num=self.state.round_num,
            event_type=event_type,
            player_id=player_id,
            details=details
        )
        self.state.history.append(event)

    @property
    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.state.phase == GamePhase.GAME_OVER
