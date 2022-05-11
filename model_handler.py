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

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from os import path
from datetime import datetime
import sqlite3
from collections import deque
import numpy as np
from random import sample

from config_server import *

def load_best_model():
    if not path.isfile(BEST_MODEL_PATH):
        return None
    else: 
        model = keras.models.load_model(BEST_MODEL_PATH)
        return model

def load_model(timestamp):
    return keras.models.load_model(path.join(MODELS_HISTORY_DIR, timestamp + ".h5"))

def load_parent_model(timestamp):
    con = sqlite3.connect(MODELS_LOG_PATH)
    cur = con.cursor()
    cur.execute('''
        SELECT parent_timestamp
        FROM training_log
        where timestamp = ?
    ''', (timestamp,))
    parent = cur.fetchall()[0][0]
    cur.close()
    con.close()
    return load_model(parent)

def load_newest_model():
    con = sqlite3.connect(MODELS_LOG_PATH)
    cur = con.cursor()
    cur.execute('''
        SELECT max(timestamp)
        FROM training_log
    ''')
    timestamp = cur.fetchall()[0][0]
    cur.close()
    con.close()

    return load_model(timestamp)

def save_model(model, num_wins_parent=None):
    def log_into_db(timestamp, num_wins_parent):
        con = sqlite3.connect(MODELS_LOG_PATH)
        cur = con.cursor()
        cur.execute('''
            SELECT max(timestamp)
            FROM training_log
            WHERE current_best = true
        ''')
        result = cur.fetchall()
        parent = None if len(result) == 0 else result[0][0]
        isBest = True if parent is None else num_wins_parent > 0
        cur.execute('''
            INSERT INTO training_log(timestamp, parent_timestamp,wins_against_parent, current_best, episodes, mtcs, cpuct) 
            VALUES (?,?,?,?,?,?,?)
        ''', (timestamp, parent))
        con.commit()
        con.close()
        return isBest
    timestamp = str(datetime.now())
    if(num_wins_parent is not None):
        isBest = log_into_db(timestamp, num_wins_parent)
        if isBest or not path.isfile(BEST_MODEL_PATH):
            model.save(BEST_MODEL_PATH)
    model.save(path.join(MODELS_HISTORY_DIR, timestamp + ".h5"))

def base_model(learning_rate = LEARNING_RATE):
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
    

class ModelHandler:
    def __init__(self, model):
        self.model = model
        self.memory = deque(maxlen = MEMORY_SIZE)
        self.fit_i = 0
        self.save_i = 0

    def predict(self, board: np.ndarray, allowed_mini_fields: np.ndarray) -> "tuple[np.ndarray, np.float64]":
        ps, v = self.model.predict((np.reshape(board, (1,9,9)), np.reshape(allowed_mini_fields, (1,9))))
        return ps[0], v[0]

    def fit(self, board: np.ndarray, allowed_mini_fields: np.ndarray, ps: np.ndarray, v: np.float64):
        self.memory.append((board, allowed_mini_fields, ps, v))
        if(len(self.memory) > MIN_MEMORY_SIZE and self.fit_i >= STATES_UNTIL_FITTING):
            self.fit_i = 0
            bs = np.empty((MEMORY_SAMPLE_SIZE, 9, 9), dtype=np.int8)
            amfs = np.empty((MEMORY_SAMPLE_SIZE, 9), dtype=np.int8)
            ps = np.empty((MEMORY_SAMPLE_SIZE, 81), dtype=np.float64)
            vs = np.empty((MEMORY_SAMPLE_SIZE), dtype=np.float64)
            for i, (b, amf, p, v) in enumerate(sample(self.memory, MEMORY_SAMPLE_SIZE)):
                bs[i] = b
                amfs[i] = amf
                ps[i] = p
                vs[i] = v
            self.model.fit((bs, amfs, ps, vs), batch_size = BATCH_SIZE)
            print("Fitting model!")
            if(self.save_i >= SAVE_INTERVAL):
                save_model(self.model)
                print("Saving model!")
                self.save_i = 0
            else:
                self.save_i += 1
            
        else:
            self.fit_i += 1



