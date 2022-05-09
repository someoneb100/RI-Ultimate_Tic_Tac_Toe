from tensorflow import keras
from config import *
from os import path
from datetime import datetime
import sqlite3

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

def save_model(model, num_wins_parent):
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
            INSERT INTO training_log(timestamp, parent_timestamp,wins_against_parent, current_best) 
            VALUES (?,?,?,?)
        ''', (timestamp, parent, num_wins_parent,isBest))
        con.commit()
        con.close()
        return isBest
    timestamp = str(datetime.now())    
    isBest = log_into_db(timestamp, num_wins_parent)
    if isBest or not path.isfile(BEST_MODEL_PATH):
        model.save(BEST_MODEL_PATH)
    model.save(path.join(MODELS_HISTORY_DIR, timestamp + ".h5"))
    
