from config import NUMBER_OF_MONTE_CARLO_SIMULATIONS, CPUCT, EPS
import math
import numpy as np
from numba import vectorize, njit, float64, int32, int8

from uttt import UltimateTicTacToe

np.seterr(invalid='ignore')

@vectorize([float64(float64)], nopython=True, target="parallel")
def get_probs(Qsa):
    return -1 if np.isnan(Qsa) or np.isinf(Qsa) else Qsa

@vectorize([float64(float64, float64, int32, int32)], nopython=True, target="parallel")
def get_Qsa(Qsa, Ps, Nsa, Ns):
    if np.isnan(Qsa):
        return CPUCT * Ps * math.sqrt(Ns + EPS)
    if np.isinf(Qsa):
        return Qsa
    return Qsa + CPUCT * Ps*math.sqrt(Ns)/(1+Nsa)

@njit([float64[:](int8[:])])
def get_initial_qsa(valid):
    res = np.repeat(np.NINF, 81)
    for i in range(len(valid)):
        res[valid[i]] = np.nan
    return res


class MonteCarlo():
    def __init__(self, model, env: UltimateTicTacToe):
        self.env = env.clone()
        self.model = model
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
    
    def getActionProb(self) -> "tuple[np.ndarray, np.float64]":
        vals = np.empty(NUMBER_OF_MONTE_CARLO_SIMULATIONS, dtype=np.float64)
        for i in range(NUMBER_OF_MONTE_CARLO_SIMULATIONS):
            vals[i] = -self.search(self.env.clone())
                
        v = vals.sum()/NUMBER_OF_MONTE_CARLO_SIMULATIONS
        qsa = self.Qsa[self.env.getCanonicalState()]
        probs = get_probs(qsa)

        probs+=1
        probs/=probs.sum()
        return probs, v
        
    def search(self, env: UltimateTicTacToe):

        s = env.getCanonicalState()
        
        # ako je novootkriveno stanje pozivamo model
        if s not in self.Ps:
            ps, v = self.model.predict([np.reshape(self.env.board, (1,9,9)), np.reshape(self.env.get_categorical_allowed_field(), (1,9))], verbose = 0)
            self.Ps[s], v = ps[0], v[0]
            self.Qsa[s] = get_initial_qsa(env.get_valid_actions())
            self.Nsa[s] = np.zeros(81, dtype=np.int32)
            return -v
          
        #biramo sledecu akciju


        a = np.nanargmax(get_Qsa(self.Qsa[s], self.Ps[s], self.Nsa[s], self.Ns.get(s, 0)))

        
        #"odigraj" sledeci potez
        reward, _ = env.play(a)
        env.flip()
        
        v = reward if env.done else self.search(env)

        #podesi nsa ns i qsa
        self.Qsa[s][a] = v if np.isnan(self.Qsa[s][a]) else (self.Nsa[s][a]*self.Qsa[s][a] + v) / (self.Nsa[s][a] + 1)
        self.Nsa[s][a] += 1
        self.Ns[s] = self.Ns.get(s,0) + 1
        return -v
        
# Klasa za testiranje koja samo uvek za sve vraca jednake verovatnoce
class SimpleModel:
    def predict(self, *args, **kwargs):
        return [np.ones(81, dtype=np.float64)/81], [0]
        
        
