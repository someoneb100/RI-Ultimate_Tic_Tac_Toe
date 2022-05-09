from collections import deque
from monte_carlo import MonteCarlo
from copy import deepcopy
from config import MEMORY_SIZE, MIN_MEMORY_SIZE, MEMORY_SAMPLE_SIZE, BATCH_SIZE
import random
import numpy as np
from uttt import FieldState, change_player

from config import DEBUG_LOG

class Agent:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.target_update_counter = 0 #pratimo kad je vreme da updateujemo target_model
        
    def play_action(self, training = False) -> FieldState:
        should_flip = False
        if self.env.player_turn == 2:
            should_flip = True
            self.env.flip()
        probs, v = MonteCarlo(self.model,self.env).getActionProb()

        if training :
            self.memory.append((self.env.board / 2, self.env.allowed_field / 8, probs, v))   
        if should_flip:
            self.env.flip()
        _, _, done, reward, info = self.env.play(np.argmax(probs))
        DEBUG_LOG.write(f"player {1 if self.env.player_turn == 2 else 2}\nprobs:\n{probs}\nv: {v}, info: {info}")
        if not done:
            return FieldState.EMPTY
        if reward == 0:
            return FieldState.TIE
        if reward == 1:
            return FieldState(self.env.player_turn)
        return FieldState(change_player(self.env.player_turn))
    
    def train(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        minibatch = random.sample(self.memory, MEMORY_SAMPLE_SIZE)
        boards = np.empty((MEMORY_SAMPLE_SIZE,9,9),dtype=np.float64)
        allowed_fields = np.empty((MEMORY_SAMPLE_SIZE,),dtype=np.float64)
        probs = np.empty((MEMORY_SAMPLE_SIZE,81),dtype=np.float64)
        vs = np.empty((MEMORY_SAMPLE_SIZE,),dtype=np.float64)
        for i, (board, allowed_field, prob, v) in enumerate(minibatch):
            boards[i] = board
            allowed_fields[i] = allowed_field
            probs[i] = prob
            vs[i] = v
        self.model.fit((boards,allowed_fields),(probs,vs), batch_size = BATCH_SIZE, verbose=0, shuffle=False)
        
        
        
        
    
