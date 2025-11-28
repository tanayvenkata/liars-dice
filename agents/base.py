"""Abstract base class for all Liar's Dice agents."""

from abc import ABC, abstractmethod

from game.types import PlayerView, Action, RoundResult


class Agent(ABC):
    """
    Abstract base class for all players (human or AI).

    An agent receives a view of the game state and must return an action.
    The agent should be stateless - all needed context is in PlayerView.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_action(self, view: PlayerView) -> Action:
        """
        Decide on an action given the current game state.

        Args:
            view: The player's view of the game state

        Returns:
            The action to take (BID or CALL)
        """
        pass

    @abstractmethod
    def notify_round_result(self, result: RoundResult) -> None:
        """
        Notify the agent of a round's result (for logging/learning).

        This is called after each round ends, providing full information
        about dice revealed and outcome.
        """
        pass

    def __str__(self) -> str:
        return self.name
