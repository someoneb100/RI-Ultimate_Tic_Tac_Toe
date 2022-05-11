import numpy as np

from uttt import FieldState, UltimateTicTacToe, change_player
from predictor_client import PredictorClient
from monte_carlo import MonteCarlo

class Agent:
    def __init__(self, model: PredictorClient, env: UltimateTicTacToe):
        self.model = model
        self.env = env
        
    def play_action(self, training: bool = False) -> FieldState:
        should_flip = False
        if self.env.player_turn == FieldState.SECOND.value:
            should_flip = True
            self.env.flip()
        probs, v = MonteCarlo(self.model,self.env).getActionProb()

        if training :
            self.model.train(self.env.board, self.env.get_categorical_allowed_field(), probs, v)   
        if should_flip:
            self.env.flip()
        reward, _ = self.env.play(np.argmax(probs))
        if not self.env.done:
            return FieldState.EMPTY
        if reward == 0:
            return FieldState.TIE
        if reward == 1:
            return FieldState(self.env.player_turn)
        return FieldState(change_player(self.env.player_turn))
        
        
        
        
        
    
