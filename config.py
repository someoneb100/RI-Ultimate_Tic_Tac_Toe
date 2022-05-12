from os import path

OUR_DEBUG = False
DEBUG_FILE = "debug_log.txt"

class DL:
    def __init__(self):
        if(OUR_DEBUG):
            self.f = open(DEBUG_FILE, "w")
            self.i = 0

    def __del__(self):
        if(OUR_DEBUG):
            self.f.close()
    
    def write(self, message):
        if(OUR_DEBUG):
            self.f.write(f"({self.i}) : {message}\n")
            self.i += 1
            self.f.flush()

DEBUG_LOG = DL()

WORK_DIR = path.abspath(path.curdir)
MODELS_DIR = path.join(WORK_DIR, "models")
BEST_MODEL_PATH = path.join(MODELS_DIR, "best.h5")
MODELS_HISTORY_DIR = path.join(MODELS_DIR, "history")
MODELS_LOG_PATH = path.join(MODELS_HISTORY_DIR, "log.db")

NUMBER_OF_MONTE_CARLO_SIMULATIONS = 400
CPUCT = 0.4
EPS = 1e-8

MEMORY_SIZE = 4_000
MIN_MEMORY_SIZE = 512
BATCH_SIZE = 32
MEMORY_SAMPLE_SIZE = MIN_MEMORY_SIZE // 2

NUMBER_OF_EPISODES = 130
LEARNING_RATE = 0.1
DUEL_LENGTH = 10
