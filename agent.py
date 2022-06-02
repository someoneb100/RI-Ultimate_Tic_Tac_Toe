from collections import deque
from monte_carlo import MonteCarlo
from copy import deepcopy
from config import MEMORY_SIZE, MIN_MEMORY_SIZE, MEMORY_SAMPLE_SIZE, BATCH_SIZE, EPS, RANDOM_FACTOR
from random import sample
import numpy as np
from uttt import FieldState, UltimateTicTacToe, change_player

class Agent:
    def __init__(self, model, env: UltimateTicTacToe):
        self.model = model
        self.env = env
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.r = False
        self.action = -1
        
    def play_action(self, training: bool = False) -> FieldState:
        should_flip = False
        if self.env.player_turn == FieldState.SECOND.value:
            should_flip = True
            self.env.flip()

        if(not training or not self.r or np.random.rand() > RANDOM_FACTOR):
            probs, v = MonteCarlo(self.model,self.env).getActionProb()
            if training :
                self.memory.append((self.env.board, self.env.get_categorical_allowed_field(), probs, v))   
            if should_flip:
                self.env.flip()
            # epsilon udaljena slucajnost
            probsmax = probs.max()
            choices = np.arange(len(probs))[np.abs(probs-probsmax)<=EPS]
        else:
            choices = self.env.get_valid_actions()
        self.action = np.random.choice(choices)
        reward, _ = self.env.play(self.action)
        if not self.env.done:
            return FieldState.EMPTY
        if reward == 0:
            return FieldState.TIE
        if reward == 1:
            return FieldState(self.env.player_turn)
        return FieldState(change_player(self.env.player_turn))
    
    def train(self) -> None:
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        minibatch = sample(self.memory, MEMORY_SAMPLE_SIZE)
        boards = np.empty((MEMORY_SAMPLE_SIZE,9,9),dtype=np.int8)
        allowed_fields = np.empty((MEMORY_SAMPLE_SIZE,9),dtype=np.int8)
        probs = np.empty((MEMORY_SAMPLE_SIZE,81),dtype=np.float64)
        vs = np.empty((MEMORY_SAMPLE_SIZE,),dtype=np.float64)
        for i, (board, allowed_field, prob, v) in enumerate(minibatch):
            boards[i] = board
            allowed_fields[i] = allowed_field
            probs[i] = prob
            vs[i] = v
        self.model.fit((boards,allowed_fields),(probs,vs), batch_size = BATCH_SIZE, verbose=0, shuffle=False)
        self.r = False
