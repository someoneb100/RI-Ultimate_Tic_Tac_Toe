from collections import deque
from monte_carlo import MonteCarlo
from copy import deepcopy
from config import MEMORY_SIZE, MIN_MEMORY_SIZE, MEMORY_SAMPLE_SIZE, BATCH_SIZE
import random
import numpy as np

class Agent:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.target_update_counter = 0 #pratimo kad je vreme da updateujemo target_model
        
    def play_action(self, training = False):
        should_flip = False
        if self.env.player_turn == 2:
            should_flip = True
            self.env.flip()
        probs, v = MonteCarlo(self.model,self.env).getActionProb()
        if training :
            self.memory.append((self.env.board / 2, self.env.allowed_field / 8, probs, v))   
        if should_flip:
            self.env.flip()
        self.env.play(np.argmax(probs))
    
    def train(self, terminal_state, step):
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
        env.model.fit((boards,allowed_fields),(probs,vs), batch_size = BATCH_SIZE, verbose=0, shuffle=False)
        
        
        
        
    
