from os import environ

# Configuration of PredictorClient

HOST_NAME = environ.get("HOST_NAME", "192.168.1.105")
HOST_PORT = int(environ.get("HOST_PORT", "1995"))
TRAINING_REQUEST = bytes(0)
TRAINING_RESPONSE = b"train"
PREDICT_REQUEST  = bytes(1)
PREDICT_RESPONSE = b"pdict"
RESPONSE_SIZE = 5

# Configuration of Monte Carlo

NUMBER_OF_MONTE_CARLO_SIMULATIONS = int(environ.get("NUMBER_OF_MONTE_CARLO_SIMULATIONS", "800"))
CPUCT = float(environ.get("CPUCT", "0.4"))
EPS = float(environ.get("EPSILON", 1e-8))

# Configuration of Coach

NUMBER_OF_EPISODES = int(environ.get("NUMBER_OF_EPISODES", "25"))