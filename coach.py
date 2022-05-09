from config import NUMBER_OF_EPISODES, LEARNING_RATE, DUEL_LENGTH
import model_handler
from uttt import FieldState, UltimateTicTacToe
from agent import Agent
import tensorflow as tf
try:
    from tqdm import tqdm
except:
    def tqdm(x, desc=None):
        if desc is not None:
            print(f"{desc}:")
        for i in x:
            yield i

from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

def test_model(learning_rate = LEARNING_RATE):
    board = Input(shape=(9,9), name="board_input")
    miniboard = Input(shape=(1,), name="allowed_mini_board_input")
    
    reshape_board = Reshape((9,9,1), name="reshape_board")(board)
    c1 = Conv2D(128, (3,3), strides=(3,3), activation="relu", kernel_initializer="glorot_normal", name="board_conv_1")(reshape_board)
    c2 = Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer="glorot_normal", name="board_conv_2")(c1)
    c3 = Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_normal", name="board_conv_3")(c2)
    flat = Flatten(name="flatten_board")(c3)
    
    concat = Concatenate(name="concat_board_and_allowed_mini_board")([flat, miniboard])
    d1 = Dense(256, activation="relu", kernel_initializer="glorot_normal", name="dense_1")(concat)
    d2 = Dense(128, activation="relu", kernel_initializer="glorot_normal", name="dense_2")(d1)
    d3 = Dense(64, activation="relu", kernel_initializer="glorot_normal", name="dense_3")(d2)
    
    probs = Dense(81, activation="softmax", name="output_action_probabilities")(d3)
    v = Dense(1, activation="tanh", name="output_board_value")(d3)
    
    model = Model(inputs=[board, miniboard], outputs=[probs,v])
    model.compile(loss=["categorical_crossentropy", "mean_squared_error"], optimizer=Adam(learning_rate))
    return model

class Coach:
    def __init__(self, model=None):
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
        score = 0
        for i in tqdm(range(DUEL_LENGTH), desc="Dueling"):
            self.env.reset()
            player1, player2 = (self.agent, sensei) if i%2 else (sensei, self.agent)
            while(True):
                res = player1.play_action()
                if res is not FieldState.EMPTY:
                    if res is not FieldState.TIE:
                        if i%2:
                            score += 1 if res is FieldState.FIRST else -1
                        else:
                            score += 1 if res is FieldState.SECOND else -1
                    break
                res = player2.play_action()
                if res is not FieldState.EMPTY:
                    if res is not FieldState.TIE:
                        if i%2:
                            score += 1 if res is FieldState.SECOND else -1
                        else:
                            score += 1 if res is FieldState.FIRST else -1
                    break
                
        return score


    def training_session(self, episodes = NUMBER_OF_EPISODES) -> None:
        for _ in tqdm(range(episodes), desc="Training"):
            self.env.reset()
            while(not self.env.done):
                self.agent.play_action(training=True)
            self.agent.train()

        score = self.duel()
        model_handler.save_model(self.agent.model, score)
        
if __name__ == "__main__":
    coach = Coach()
    coach.training_session()