"""Core data types for Liar's Dice."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import random


class ActionType(Enum):
    """Types of actions a player can take."""
    BID = auto()
    CALL = auto()


@dataclass(frozen=True)
class Bid:
    """
    Represents a bid in Liar's Dice.

    Attributes:
        quantity: Number of dice claimed (e.g., 3 for "three fours")
        face_value: The face value claimed (1-6)
    """
    quantity: int
    face_value: int

    def __post_init__(self):
        if not (1 <= self.face_value <= 6):
            raise ValueError(f"Face value must be 1-6, got {self.face_value}")
        if self.quantity < 1:
            raise ValueError(f"Quantity must be at least 1, got {self.quantity}")

    def __str__(self) -> str:
        face_names = {1: "ones", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes"}
        return f"{self.quantity} {face_names[self.face_value]}"

    def is_valid_raise_over(self, previous: Optional["Bid"]) -> bool:
        """Check if this bid is a valid raise over the previous bid."""
        if previous is None:
            return True
        # Must increase quantity OR face value (or both)
        return (self.quantity > previous.quantity or
                (self.quantity == previous.quantity and self.face_value > previous.face_value))


@dataclass
class Action:
    """
    Represents a player's action.

    Attributes:
        action_type: BID or CALL
        bid: The bid (required if action_type is BID)
        reasoning: Optional explanation from the player
    """
    action_type: ActionType
    bid: Optional[Bid] = None
    reasoning: Optional[str] = None

    def __post_init__(self):
        if self.action_type == ActionType.BID and self.bid is None:
            raise ValueError("BID action must include a bid")

    def __str__(self) -> str:
        if self.action_type == ActionType.CALL:
            return "CALL"
        return f"BID {self.bid}"


@dataclass
class PlayerState:
    """
    The private state for a single player.

    Attributes:
        player_id: Unique identifier (0 or 1)
        dice: List of face values for this player's dice
        dice_count: Number of dice remaining
    """
    player_id: int
    dice: list[int] = field(default_factory=list)
    dice_count: int = 5

    def roll_dice(self) -> None:
        """Roll all dice for this player."""
        self.dice = [random.randint(1, 6) for _ in range(self.dice_count)]

    def lose_die(self) -> bool:
        """
        Remove one die from this player.

        Returns:
            True if player is eliminated (no dice left)
        """
        self.dice_count -= 1
        return self.dice_count <= 0


@dataclass
class GameEvent:
    """
    Represents an event in the game history (whiteboard).

    Attributes:
        round_num: The round number
        event_type: Type of event (round_start, bid, call, round_result)
        player_id: The player who triggered the event
        details: Additional event details
    """
    round_num: int
    event_type: str
    player_id: Optional[int]
    details: str


class GamePhase(Enum):
    """Current phase of the game."""
    WAITING_TO_START = auto()
    ROUND_IN_PROGRESS = auto()
    ROUND_ENDED = auto()
    GAME_OVER = auto()


@dataclass
class GameState:
    """
    The complete game state.

    Attributes:
        players: List of PlayerState for each player
        current_player: Index of player whose turn it is
        current_bid: The current bid on the table (None if round just started)
        round_num: Current round number
        phase: Current game phase
        history: List of all game events (the whiteboard)
        winner: Player index of winner (None if game ongoing)
    """
    players: list[PlayerState] = field(default_factory=list)
    current_player: int = 0
    current_bid: Optional[Bid] = None
    round_num: int = 1
    phase: GamePhase = GamePhase.WAITING_TO_START
    history: list[GameEvent] = field(default_factory=list)
    winner: Optional[int] = None

    @property
    def total_dice(self) -> int:
        """Total dice in play across all players."""
        return sum(p.dice_count for p in self.players)

    @property
    def opponent_dice_count(self) -> dict[int, int]:
        """Map of player_id to their dice count."""
        return {p.player_id: p.dice_count for p in self.players}


@dataclass
class PlayerView:
    """
    What a player can see about the game state.
    This is the information passed to agents - they never see opponent dice.

    Attributes:
        player_id: This player's ID
        own_dice: This player's current dice values
        opponent_dice_count: Number of dice the opponent has
        current_bid: The current bid (None if this player opens)
        round_num: Current round number
        total_dice: Total dice in play
        history: Full game history (whiteboard)
    """
    player_id: int
    own_dice: list[int]
    opponent_dice_count: int
    current_bid: Optional[Bid]
    round_num: int
    total_dice: int
    history: list[GameEvent]


@dataclass
class RoundResult:
    """
    Result of a round after a call is made.

    Attributes:
        caller_id: Player who called
        bidder_id: Player who made the challenged bid
        challenged_bid: The bid that was challenged
        actual_count: Actual count of the bid's face value
        all_dice: All dice revealed {player_id: [dice]}
        loser_id: Player who lost the round
        bid_was_valid: True if the bid was met or exceeded
        bid_history: Full history of bids made this round [(player_id, bid), ...]
    """
    caller_id: int
    bidder_id: int
    challenged_bid: Bid
    actual_count: int
    all_dice: dict[int, list[int]]
    loser_id: int
    bid_was_valid: bool
    bid_history: list[tuple[int, Bid]] = field(default_factory=list)
