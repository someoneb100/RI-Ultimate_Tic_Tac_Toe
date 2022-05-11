import numpy as np
from socket import socket, AF_INET, SOCK_STREAM
from config_client import HOST_NAME, HOST_PORT, TRAINING_REQUEST, TRAINING_RESPONSE, PREDICT_REQUEST, PREDICT_RESPONSE, RESPONSE_SIZE


class PredictorClient:
    def __init__(self, host:str = HOST_NAME, port:int = HOST_PORT):
        self.host = host
        self.port = port

    def __del__(self):
        self.socket.close()

    def predict(self, board: np.ndarray, allowed_miniboard: np.ndarray) -> "tuple[np.ndarray, np.float32]":
        try:
            self.socket = socket(AF_INET, SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        except:
            print("Failed to connect to UTTT server!")
            exit(1)
        with self.socket:
            self.socket.sendall(PREDICT_REQUEST)
            package = board.tobytes() + allowed_miniboard.tobytes()
            self.socket.sendall(package)
            package = self.socket.recv(81*8 + 8)
            ps = np.frombuffer(package, dtype=np.float32, count=81)
            v  = np.frombuffer(package, dtype=np.float32, offset=81*4)[0]
            return ps, v

    def train(self, board: np.ndarray, allowed_miniboard: np.ndarray, ps: np.ndarray, v: np.float64) -> None:
        try:
            self.socket = socket(AF_INET, SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        except:
            print("Failed to connect to UTTT server!")
            exit(1)
        with self.socket:
            self.socket.sendall(TRAINING_REQUEST)
            package = board.tobytes() + allowed_miniboard.tobytes() + ps.tobytes() + v.tobytes()
            self.socket.sendall(package)
