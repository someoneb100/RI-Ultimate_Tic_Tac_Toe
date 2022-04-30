from keras import models
from config import *
from os import path
from datetime import datetime
import sqlite3

def load_best_model():
    if not path.isdir(BEST_MODEL_DIR):
        return None
    else: 
        model = keras.models.load_model(BEST_MODEL_DIR)
        return model

def load_model(timestamp):
    model = keras.models.load_model(path.join(MODELS_HISTORY_DIR,timestamp))
    return model

def save_model(model, num_wins_parent, num_wins_partner):
    def log_into_db(timestamp, num_wins_parent, num_wins_partner):
        con = sqlite3.connect(MODELS_LOG_PATH)
        cur = con.cursor()
        cur.execute('''
            SELECT max(timestamp)
            FROM training_log
            WHERE current_best = true
        ''')
        result = cur.fetch_all()
        parent = None if len(result) == 0 else result[0][0]
        isBest = True if parent is None else num_wins_parent > 0
        cur.execute('''
            INSERT INTO training_log(timestamp,parent_timestamp,wins_against_partner,wins_against_parent, current_best) 
            VALUES (?,?,?,?,?)
        ''', (timestamp, parent, num_wins_partner, num_wins_parent,isBest))
        con.commit()
        con.close()
        return isBest
    timestamp = str(datetime.now())    
    isBest = log_into_db(timestamp, num_wins_parent, num_wins_partner);
    if isBest  or not path.isdir(BEST_MODEL_DIR):
        model.save(BEST_MODEL_DIR)
    model.save(path.join(MODELS_HISTORY_DIR, timestamp))
    
