"""
Deterministic agent for Liar's Dice.

Uses optimal thresholds discovered via 32.4M game round-robin tournament (Nov 2025).

Decision Logic:
    1. If best raise has P >= safe_raise_threshold (15%): RAISE
    2. Elif current bid has P < call_threshold (35%): CALL
    3. Else: RAISE (forced bluff to survive)

Optimal Thresholds:
    - call_threshold = 0.35 (call if P(bid true) < 35%)
    - safe_raise_threshold = 0.15 (raise if P >= 15%)
    - Beats 77/80 opponent configurations
    - 56.5% average win rate across all opponents

Personalities:
    Different threshold combinations for gameplay variety:
    - MANIAC: Aggressive bluffing and rarely calls
    - AGGRESSIVE: Bluffs freely and calls everything
    - PASSIVE: Plays honest, rarely calls
    - CALLING_STATION: Plays honest, calls everything
    - OPTIMAL: Best performing thresholds (default)
"""

from game.types import PlayerView, Action, ActionType, Bid, RoundResult
from game.prompts_v2 import (
    binomial_prob_at_least,
    get_raise_options,
    DEFAULT_CALL_THRESHOLD,
    DEFAULT_SAFE_RAISE_THRESHOLD,
)
from agents.base import Agent


# Personality presets: (call_threshold, safe_raise_threshold)
# These define distinct playing styles for gameplay variety
PERSONALITIES = {
    "MANIAC": (0.10, 0.10),           # Bluffs freely, never calls
    "AGGRESSIVE": (0.50, 0.10),       # Bluffs freely, calls everything
    "PASSIVE": (0.10, 0.50),          # Plays honest, never calls
    "CALLING_STATION": (0.50, 0.50),  # Plays honest, calls everything
    "OPTIMAL": (0.35, 0.15),          # Best performing (tournament winner)
}


def create_personality_agent(personality: str, name: str = None, quiet: bool = True) -> Agent:
    """Create a deterministic agent with a specific personality.

    Args:
        personality: One of MANIAC, AGGRESSIVE, PASSIVE, CALLING_STATION, OPTIMAL
        name: Display name (defaults to personality name)
        quiet: If True, suppress debug output

    Returns:
        Agent instance with the specified personality
    """
    personality = personality.upper()

    if personality not in PERSONALITIES:
        raise ValueError(f"Unknown personality: {personality}. "
                        f"Choose from: {list(PERSONALITIES.keys())}")

    call_th, raise_th = PERSONALITIES[personality]
    return DeterministicV2Agent(
        name or personality.title().replace("_", " "),
        call_threshold=call_th,
        safe_raise_threshold=raise_th,
        quiet=quiet
    )


class DeterministicV2Agent(Agent):
    """
    Deterministic agent using optimal probability-based strategy.

    Decision logic:
    1. If best raise has P >= safe_raise_threshold: RAISE (safe)
    2. Elif current bid has P < call_threshold: CALL (likely bluff)
    3. Else: RAISE to best option (forced bluff to survive)

    No randomness, no LLM - pure math with optimal thresholds.
    """

    def __init__(
        self,
        name: str,
        call_threshold: float = DEFAULT_CALL_THRESHOLD,
        safe_raise_threshold: float = DEFAULT_SAFE_RAISE_THRESHOLD,
        quiet: bool = True,
    ):
        super().__init__(name)
        self.call_threshold = call_threshold
        self.safe_raise_threshold = safe_raise_threshold
        self.quiet = quiet

    def get_action(self, view: PlayerView) -> Action:
        """Make a deterministic decision based on probabilities."""

        # Opening bid - bid what we have most of
        if view.current_bid is None:
            return self._make_opening_bid(view)

        # Calculate probabilities
        face_counts = {i: view.own_dice.count(i) for i in range(1, 7)}
        bid_face = view.current_bid.face_value
        bid_qty = view.current_bid.quantity
        your_count = face_counts.get(bid_face, 0)
        need_from_opp = max(0, bid_qty - your_count)

        prob_bid_true = binomial_prob_at_least(view.opponent_dice_count, need_from_opp)

        # Get best raise option - filter out impossible bids
        options = get_raise_options(view.current_bid, view.own_dice, view.opponent_dice_count)
        options = [o for o in options if o["qty"] <= view.total_dice]
        best_raise = options[0] if options else None
        best_raise_prob = best_raise["prob"] if best_raise else 0

        # If no valid raises available, must call
        if best_raise is None:
            return Action(
                ActionType.CALL,
                reasoning="No valid raise options - must call"
            )

        # Decision logic
        if best_raise_prob >= self.safe_raise_threshold:
            # Can raise safely
            action = Action(
                ActionType.BID,
                bid=Bid(best_raise["qty"], best_raise["face"]),
                reasoning=f"Safe raise: P={best_raise_prob:.0%} >= {self.safe_raise_threshold:.0%}"
            )
        elif prob_bid_true < self.call_threshold:
            # Bid is likely false - call
            action = Action(
                ActionType.CALL,
                reasoning=f"Call bluff: P={prob_bid_true:.0%} < {self.call_threshold:.0%}"
            )
        else:
            # Forced to bluff
            action = Action(
                ActionType.BID,
                bid=Bid(best_raise["qty"], best_raise["face"]),
                reasoning=f"Forced bluff: P={best_raise_prob:.0%}"
            )

        if not self.quiet:
            print(f"[{self.name}] {action.reasoning} -> {action}")

        return action

    def _make_opening_bid(self, view: PlayerView) -> Action:
        """Make opening bid based on what we have most of."""
        face_counts = {i: view.own_dice.count(i) for i in range(1, 7)}

        # Find face we have most of
        best_face = max(face_counts, key=face_counts.get)
        best_count = face_counts[best_face]

        # Bid what we have (conservative opening)
        return Action(
            ActionType.BID,
            bid=Bid(best_count, best_face),
            reasoning=f"Opening with {best_count} {best_face}s (have them)"
        )

    def notify_round_result(self, result: RoundResult) -> None:
        """No-op for stateless agent."""
        pass
