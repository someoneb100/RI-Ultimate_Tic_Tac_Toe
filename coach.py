from config import NUMBER_OF_EPISODES, LEARNING_RATE, DUEL_LENGTH, GAMES_PER_TRAINING, TRAINING_EPISODES
import model_handler
from uttt import FieldState, UltimateTicTacToe
from agent import Agent
try:
    from tqdm import trange
except:
    def trange(x, desc=None):
        if desc is not None:
            print(f"{desc}:")
        for i in x:
            yield i

from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def test_model(learning_rate = LEARNING_RATE):
    board = Input(shape=(9,9), name="board_input")
    miniboard = Input(shape=(9,), name="allowed_mini_board_input")
    
    reshape_board = Reshape((9,9,1), name="reshape_board")(board)
    c1 = Conv2D(256, (3,3), strides=(3,3), activation="relu", kernel_initializer="glorot_normal", name="board_conv_1")(reshape_board)
    c2 = Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer="glorot_normal", name="board_conv_2")(c1)
    c3 = Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_normal", name="board_conv_3")(c2)
    flat = Flatten(name="flatten_board")(c3)
    
    concat = Concatenate(name="concat_board_and_allowed_mini_board")([flat, miniboard])
    d1 = Dense(512, activation="relu", kernel_initializer="glorot_normal", name="dense_1")(concat)
    d2 = Dense(256, activation="relu", kernel_initializer="glorot_normal", name="dense_2")(d1)
    d3 = Dense(128, activation="relu", kernel_initializer="glorot_normal", name="dense_3")(d2)
    
    probs = Dense(81, activation="softmax", name="output_action_probabilities")(d3)
    v = Dense(1, activation="tanh", name="output_board_value")(d3)
    
    model = Model(inputs=[board, miniboard], outputs=[probs,v])
    model.compile(loss=["categorical_crossentropy", "mean_squared_error"], optimizer=Adam(learning_rate), metrics = ["accuracy"])
    return model

class Coach:
    def __init__(self, model=None):
        if model is None:
            model = model_handler.load_best_model()
        elif type(model) == type(""):
            model = model_handler.load_model(model)
        if model is None:
            model = test_model()
        self.env = UltimateTicTacToe()
        self.agent = Agent(model, self.env)

    def duel(self) -> int:
        sensei = model_handler.load_best_model()
        if sensei is None:
            return 0
        sensei = Agent(sensei, self.env)
        sensei.score, self.agent.score, tie = 0, 0, 0
        with trange(DUEL_LENGTH) as pbar:
            for i in pbar:
                pbar.set_description(f"Training: win={self.agent.score}; loss={sensei.score}; tie={tie}")
                self.env.reset()
                player1, player2 = (self.agent, sensei) if i%2 else (sensei, self.agent)
                while(True):
                    res = player1.play_action()
                    if res is not FieldState.EMPTY:
                        if res is not FieldState.TIE:
                            if res is FieldState.FIRST:
                                player1.score += 1
                            else:
                                player2.score += 1
                        else:
                            tie += 1
                        break
                    res = player2.play_action()
                    if res is not FieldState.EMPTY:
                        if res is not FieldState.TIE:
                            if res is FieldState.SECOND:
                                player2.score += 1
                            else:
                                player1.score += 1
                        else:
                            tie += 1
                        break
        return self.agent.score - sensei.score


    def training_session(self, episodes = NUMBER_OF_EPISODES) -> None:
        for i in trange(episodes, desc="Training"):
            self.env.reset()
            while(not self.env.done):
                self.agent.play_action(training=True)
            if(i%GAMES_PER_TRAINING) == (GAMES_PER_TRAINING - 1):
                for _ in range(TRAINING_EPISODES):
                    self.agent.train()

        score = self.duel()
        model_handler.save_model(self.agent.model, score)
        
if __name__ == "__main__":
    coach = Coach(model_handler.load_newest_model())
    coach.training_session()