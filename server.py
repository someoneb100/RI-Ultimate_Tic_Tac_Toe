from cgitb import handler
import socket
import numpy as np

from config_server import *
import model_handler

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    res = s.getsockname()[0]
    s.close()
    return res

HOST_NAME = get_ip_address()

class TFServer:
    def __init__(self, model:model_handler.ModelHandler ,host: str = HOST_NAME, port: int = HOST_PORT):
        self.model = model
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.bind(("", port))
            self.s.listen(NUMBER_OF_CONNECTIONS)
            msg = f"# Listening on address {host}:{port}..."
            print("#"*len(msg))
            print("#")
            print(msg)
            print("#")
            print("#"*len(msg))
        except:
            print("Failed to bind a connection")
            exit(1)
        
    def __del__(self):
        self.s.close()

    def listen(self):
        while True:
            conn, _ = self.s.accept()
            with conn:
                request = conn.recv(1)
                if(request == PREDICT_REQUEST):
                    package = conn.recv(9*9 + 9)
                    board = np.reshape(np.frombuffer(package, dtype=np.int8, count=9*9), (9,9))
                    allowed_mini_fields = np.frombuffer(package, dtype=np.int8, offset=9*9)
                    ps, v = self.model.predict(board, allowed_mini_fields)
                    package = ps.tobytes() + v.tobytes()
                    conn.send(package)
                elif(request == TRAINING_REQUEST):
                    package = conn.recv(9*9 + 9 + 4*9*9 + 4)
                    board = np.reshape(np.frombuffer(package, dtype=np.int8, count=9*9), (9,9))
                    amf = np.frombuffer(package, dtype=np.int8, offset=9*9, count=9)
                    ps = np.frombuffer(package, dtype=np.float64, offset=9*9+9, count=9*9)
                    v = np.frombuffer(package, dtype=np.float64, offset=9*9 + 9 + 4*9*9)[0]
                    self.model.fit(board, amf, ps, v)
                else:
                    conn.send(b'error')

if __name__ == "__main__":
    handler = model_handler.ModelHandler(model_handler.load_newest_model())
    server = TFServer(handler)
    try:
        server.listen()
    except KeyboardInterrupt:
        print("Server shutting down...")
        r = input("Save? [y/n] ")
        if(r == "y"):
            model_handler.save_model(handler.model)