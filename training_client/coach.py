from tqdm import tqdm

from config_client import NUMBER_OF_EPISODES
from uttt import UltimateTicTacToe
from agent import Agent
from predictor_client import PredictorClient

class Coach:
    def __init__(self, model: PredictorClient):
        self.env = UltimateTicTacToe()
        self.agent = Agent(model, self.env)

    def training_session(self, episodes = NUMBER_OF_EPISODES) -> None:
        for _ in tqdm(range(episodes), desc="Training"):
            self.env.reset()
            while(not self.env.done):
                self.agent.play_action(training=True)
        
if __name__ == "__main__":
    client = PredictorClient()
    coach = Coach(client)
    try:
        coach.training_session()
    except KeyboardInterrupt:
        print("Ended execution with KeyBoard Interrupt")
    del client