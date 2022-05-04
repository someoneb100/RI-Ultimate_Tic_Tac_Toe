from os import path

WORK_DIR = path.abspath(path.curdir)
MODELS_DIR = path.join(WORK_DIR, "models")
BEST_MODEL_DIR = path.join(MODELS_DIR, "best")
MODELS_HISTORY_DIR = path.join(MODELS_DIR, "history")
MODELS_LOG_PATH = path.join(MODELS_HISTORY_DIR, "log.db")

NUMBER_OF_MONTE_CARLO_SIMULATIONS = 600
CPUCT = 0,4
EPS = 1e-8
