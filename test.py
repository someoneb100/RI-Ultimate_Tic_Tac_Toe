from coach import Coach
from monte_carlo import SimpleModel
import model_handler

duel_length = 10
coach = Coach(model_handler.load_newest_model())
coach.duel(sensei=SimpleModel(), duel_length=duel_length)