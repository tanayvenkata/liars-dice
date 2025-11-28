# Architecture Overview

This document describes the system architecture of Liar's Dice, designed for developers who want to understand the codebase or build custom agents.

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         play.py (CLI)                               │
│    - Game mode selection                                            │
│    - Agent creation                                                 │
│    - Game loop orchestration                                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GameEngine (game/engine.py)                      │
│    - State management (GameState)                                   │
│    - Rule enforcement                                               │
│    - Action validation & application                                │
│    - Round lifecycle                                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│      Agent Interface      │   │        Data Types (game/types.py) │
│    (agents/base.py)       │   │    - Bid, Action, PlayerView      │
│                           │   │    - GameState, RoundResult       │
│    - get_action()         │   │    - GameEvent, GamePhase         │
│    - notify_round_result()│   └───────────────────────────────────┘
└───────────┬───────────────┘
            │
    ┌───────┴───────────────┐
    ▼                       ▼
┌─────────┐          ┌──────────────┐
│  Human  │          │ Deterministic│
│  Agent  │          │    Agent     │
└─────────┘          └──────┬───────┘
                            │
                            ▼
                    ┌────────────────┐
                    │  prompts_v2.py │
                    │ (probability)  │
                    └────────────────┘
```

## Core Components

### 1. Game Engine (`game/engine.py`)

The engine is the authoritative source for game state. It:
- Manages the complete `GameState` object
- Validates all actions before applying them
- Enforces game rules (valid bids, when to call)
- Handles round lifecycle (roll → bid → call → resolve → repeat)

**Key principle**: The engine is agent-agnostic. It doesn't know or care how players make decisions.

```python
# Round lifecycle
engine._start_new_round()     # Roll dice, clear bid
engine.get_player_view(id)    # Generate PlayerView for agent
engine.validate_action(...)   # Check action legality
engine.apply_action(...)      # Update state, return RoundResult if CALL
```

### 2. Data Types (`game/types.py`)

Immutable data classes that flow through the system:

| Type | Purpose |
|------|---------|
| `Bid` | A claim (quantity, face_value) - frozen/immutable |
| `Action` | Player's move (BID+bid or CALL) with optional reasoning |
| `PlayerView` | What an agent sees - own dice, opponent count, current bid |
| `GameState` | Complete state - all dice, history, phase |
| `RoundResult` | Revealed after CALL - all dice, who won |

**Information hiding**: Agents receive `PlayerView`, never `GameState`. They cannot see opponent's dice.

### 3. Agent Interface (`agents/base.py`)

All agents implement two methods:

```python
class Agent(ABC):
    def get_action(self, view: PlayerView) -> Action:
        """Decide BID or CALL given current state."""
        pass

    def notify_round_result(self, result: RoundResult) -> None:
        """Learn from revealed dice after each round."""
        pass
```

## Deterministic Agent (`agents/deterministic_agent.py`)

Pure math probability-based decisions using binomial calculations.

### Optimal Thresholds (Tournament Winner)

Discovered via 32.4M game round-robin tournament:

```python
DEFAULT_CALL_THRESHOLD = 0.35      # Call if P(bid true) < 35%
DEFAULT_SAFE_RAISE_THRESHOLD = 0.15  # Raise if P >= 15%
```

### Decision Logic

```
1. If best raise has P >= 15%: RAISE (safe)
2. Elif current bid has P < 35%: CALL (likely bluff)
3. Else: RAISE (forced bluff to survive)
```

### Core Math (`game/prompts_v2.py`)

```python
# P(opponent has at least k dice of a face) = sum of binomial probabilities
def binomial_prob_at_least(n, k, p=1/6):
    return sum(C(n,i) * p**i * (1-p)**(n-i) for i in range(k, n+1))
```

### Personality Presets

Different threshold combinations for gameplay variety:

```python
PERSONALITIES = {
    "MANIAC": (0.10, 0.10),           # Bluffs freely, never calls
    "AGGRESSIVE": (0.50, 0.10),       # Bluffs freely, calls everything
    "PASSIVE": (0.10, 0.50),          # Plays honest, never calls
    "CALLING_STATION": (0.50, 0.50),  # Plays honest, calls everything
    "OPTIMAL": (0.35, 0.15),          # Tournament winner
}
```

## Extension Points

### Adding a Custom Agent

1. Create a new file in `agents/`:

```python
from agents.base import Agent
from game.types import PlayerView, Action, ActionType, Bid, RoundResult

class MyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        # Your state here

    def get_action(self, view: PlayerView) -> Action:
        # view.own_dice - your dice
        # view.opponent_dice_count - their count
        # view.current_bid - bid to beat (None = you open)
        # view.total_dice - total in play

        # Return BID or CALL
        if should_call(view):
            return Action(ActionType.CALL)
        return Action(ActionType.BID, bid=Bid(quantity, face))

    def notify_round_result(self, result: RoundResult) -> None:
        # result.all_dice - revealed dice
        # result.bid_history - all bids this round
        # Use for learning/adaptation
        pass
```

2. Test against built-in agents:

```python
from game.engine import GameEngine
from agents.deterministic_agent import create_personality_agent

my_agent = MyAgent("MyBot")
opponent = create_personality_agent("OPTIMAL")

engine = GameEngine(starting_dice=5)
engine.initialize_game(num_players=2)
engine.start_game()

while not engine.is_game_over:
    current_player = engine.state.current_player
    agent = my_agent if current_player == 0 else opponent
    view = engine.get_player_view(current_player)
    action = agent.get_action(view)
    result = engine.apply_action(current_player, action)
    if result:
        my_agent.notify_round_result(result)
        opponent.notify_round_result(result)
```

## Statistical Utilities (`stats_utils.py`)

Helper functions for experiment analysis:

| Function | Purpose |
|----------|---------|
| `wilson_ci(wins, total)` | 95% confidence interval for win rate |
| `z_test_proportions(w1, n1, w2, n2)` | Compare two agents' win rates |
| `binomial_test(k, n, p)` | Test if rate differs from baseline |

## File Reference

```
liars-dice/
├── game/
│   ├── engine.py          # GameEngine - state & rules
│   ├── types.py           # Core data types
│   └── prompts_v2.py      # Probability calculations
├── agents/
│   ├── base.py            # Agent ABC
│   ├── human.py           # CLI input
│   └── deterministic_agent.py  # AI agent
├── results/
│   └── tournament_heatmap.png  # Tournament results
├── play.py                # Interactive CLI
├── visualize.py           # Tournament visualization
├── test_deterministic.py  # Test harness
└── stats_utils.py         # Statistical utilities
```
