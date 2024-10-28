import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
import pandas as pd


#States in tennis game
class State:
    def __init__(self, player_points: int, opponent_points: int, serve: int):
        self.player_points = player_points # 0=0 | 1=15 | 2=30/disadvantage | 3=40/advantage | 4-win
        self.opponent_points = opponent_points  # 0=0 | 1=15 | 2=30/disadvantage | 3=40/advantage  | 4-opponent win
        self.serve = serve # 1=first serve | 2=second serve
    def __eq__(self, other):
        if isinstance(other, State):
            return (self.player_points == other.player_points and
                    self.opponent_points == other.opponent_points and
                    self.serve == other.serve)
        return False
    def __hash__(self):
        return hash((self.player_points, self.opponent_points, self.serve))
    def __str__(self):
        if self.player_points == 0:
            true_player_points = 0
        if self.player_points == 1:
            true_player_points = 15
        if self.player_points == 2:
            true_player_points = 30
        if self.player_points == 3:
            true_player_points = 40
        if self.opponent_points == 0:
            true_opponent_points = 0
        if self.opponent_points == 1:
            true_opponent_points = 15
        if self.opponent_points == 2:
            true_opponent_points = 30
        if self.opponent_points == 3:
            true_opponent_points = 40
        if self.player_points == 4:
            return f"[WIN]"
        if self.opponent_points == 4:
            return f"[LOSE]"
        if self.player_points == 3 and self.opponent_points == 3:
            return f"[DEUCE|{self.serve}]"
        if self.player_points == 3 and self.opponent_points == 2:
            return f"[ADV|{self.serve}]"
        if self.player_points == 2 and self.opponent_points == 3:
            return f"[DIS|{self.serve}]"
        return f"[{true_player_points}|{true_opponent_points}|{self.serve}]"

states = set()
winning_states = set()
losing_states = set()
for player_points in range(0,4+1):
    for opponent_points in range(0,4+1):
        if not (player_points==4 and opponent_points==4) and not (player_points==4 and opponent_points==3) and not (player_points==3 and opponent_points==4):
            states.add(State(player_points, opponent_points, 1))
            if player_points==4:
                winning_states.add(State(player_points, opponent_points, 1))
            if opponent_points==4:
                losing_states.add(State(player_points, opponent_points, 1))
            if not (player_points==4 or opponent_points==4):
                states.add(State(player_points, opponent_points, 2))

#Transition probabilities
def getProb(state_from: State, state_to: State, p: float, q: float, winning_states: set[State], losing_states: set[State]) -> float:    
    #Terminal states
    if (state_from in winning_states or state_from in losing_states) and state_from == state_to: 
        return 1
    #Win point on either serve
    if state_to.player_points == state_from.player_points + 1 and state_to.opponent_points == state_from.opponent_points and state_to.serve == 1: 
        return p * q
    #Win point on either serve on deuce
    if state_to.player_points == 3 and state_from.player_points == 3 and state_to.opponent_points == 2 and state_from.opponent_points == 3 and state_to.serve == 1: 
        return p * q
    # first serve in net
    if state_to.player_points == state_from.player_points and state_to.opponent_points == state_from.opponent_points and state_to.serve == state_from.serve + 1: 
        return 1 - p
    # Lose point on first serve
    if state_to.player_points == state_from.player_points and state_to.opponent_points == state_from.opponent_points + 1 and state_from.serve == 1 and state_to.serve == 1: 
        return p * (1 - q)
    # Lose point on first serve at deuce
    if state_to.player_points == 2 and state_from.player_points == 3 and state_to.opponent_points == 3 and state_from.opponent_points == 3 and state_to.serve == 1:
        return p * (1 - q)
    # Lose point on second serve
    if state_to.player_points == state_from.player_points and state_to.opponent_points == state_from.opponent_points + 1 and state_from.serve == 2 and state_to.serve == 1: 
        return 1 - p * q
    # Lose point on second serve at deuce
    if state_to.player_points == 2 and state_from.player_points == 3 and state_to.opponent_points == 3 and state_from.opponent_points == 3 and state_from.serve == 2 and state_to.serve == 1:
        return 1 - p * q
    return 0

#Actions            
class Serve(Enum):
    FAST = 0
    SLOW = 1

# Landing probability
pF = 0.55 # fast serve
pS = 0.95  # slow serve
# Winning probability
qF = 0.6  # fast serve
qS = 0.4  # slow serve


#Value iteration
value: dict[State, float] = {}

max_diff = np.inf
it = 0
while max_diff > 1e-8 and it < 1000:
    value_old = value.copy()
    for state_from in states:
        max_value = -np.inf
        for action in {Serve.FAST, Serve.SLOW}:
            action_value = 0
            if action == Serve.FAST:
                p: float = pF
                q: float  = qF
            else: # action == Serve.SLOW
                p: float  = pS
                q: float  = qS
            for state_to in states:
                prob = getProb(state_from, state_to, p, q, winning_states, losing_states) #Probabilities
                rew = 0
                if state_to in winning_states and not state_from in winning_states:
                    rew = 1
                action_value += prob * (rew + 1*value.get(state_to, 0))
            if action_value > max_value:
                max_value = action_value
        value[state_from] = max_value
    
    #termination condition
    max_diff = 0
    for state in states:
        max_diff = max(max_diff, np.abs(value.get(state, 10) - value_old.get(state, 0)))
    it += 1
    #print(f"Iteration {it} max_diff: {max_diff}")

optimal_serve: dict[State, Serve] = {}

for state_from in states:
        max_value = -np.inf
        for action in {Serve.FAST, Serve.SLOW}:
            action_value = 0
            if action == Serve.FAST:
                p: float = pF
                q: float  = qF
            else: # action == Serve.SLOW
                p: float  = pS
                q: float  = qS
            for state_to in states:
                prob = getProb(state_from, state_to, p, q, winning_states, losing_states) #Probabilities
                rew = 0
                if state_to in losing_states:
                    rew = -1
                action_value += prob * (rew + 0.9*value.get(state_to, 0))
            if action_value > max_value:
                max_value = action_value
                optimal_serve[state_from] = action

for state in states:
    if state in winning_states:
        value[state] = 1

for state in states:
    print(f"{state} {optimal_serve.get(state, None)} | Winning Probabibility: {value.get(state, 0)}")


