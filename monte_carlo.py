from config import NUMBER_OF_MONTE_CARLO_SIMULATIONS, CPUCT, EPS
import math
import numpy as np
from numba import vectorize, float64, int32

np.seterr(invalid='ignore')

@vectorize([float64(float64)], nopython=True, target="parallel")
def get_probs(Qsa):
    return -1 if np.isnan(Qsa) or np.isinf(Qsa) else Qsa

@vectorize([float64(float64, float64, int32, int32)], nopython=True, target="parallel")
def get_Qsa(Qsa, Ps, Nsa, Ns):
    return CPUCT * Ps * math.sqrt(Ns + EPS) if np.isnan(Qsa) else Qsa + CPUCT * Ps*math.sqrt(Ns)/(1+Nsa)


class MonteCarlo():
    def __init__(self, model, env):
        self.env = env.clone()
        self.model = model
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
    
    def getActionProb(self):
        
        vals = []
        for i in range(NUMBER_OF_MONTE_CARLO_SIMULATIONS):
            res = self.search(self.env.clone())
            if res is not None:
                vals.append(res)
                
        v = sum(vals)/len(vals)
        qsa = self.Qsa[self.env.getCanonicalState()]
        probs = get_probs(qsa)

        probs+=1
        probs/=probs.sum()
        return probs, v
        
    def search(self, env):

        s = env.getCanonicalState()
        
        # ako je novootkriveno stanje pozivamo model
        if s not in self.Ps:
            ps, v = self.model.predict([np.reshape(self.env.board/2, (1,9,9)), np.array([self.env.allowed_field/8])])
            self.Ps[s], v = ps[0], v[0]
            self.Qsa[s] = np.repeat(np.nan, 81)
            self.Nsa[s] = np.zeros(81, dtype=np.int32)
            return -v
          
        #biramo sledecu akciju


        a = np.nanargmax(get_Qsa(self.Qsa[s], self.Ps[s], self.Nsa[s], self.Ns.get(s, 0)))

        
        #"odigraj" sledeci potez
        _, _, reward, done, info = env.play(a)
        env.flip()
        
        #ako bi ovom akcijom usli u nevalidno stanje
        if(info.startswith('Invalid move:')):
            self.Qsa[s][a] = np.NINF
            self.Nsa[s][a] = 1
            self.Ns[s] = self.Ns.get(s,0)
            return None
        elif done: #ako bi ovom akcijom usli u zavrsno stanje
            v = reward
        else:
            #rekurzivni poziv
            v = self.search(env)
        if v is not None:
        #podesi nsa ns i qsa
            self.Qsa[s][a] = v if np.isnan(self.Qsa[s][a]) else (self.Nsa[s][a]*self.Qsa[s][a] + v) / (self.Nsa[s][a] + 1)
            self.Nsa[s][a] += 1
            self.Ns[s] = self.Ns.get(s,0) + 1
            return -v
        
        return None
        
        
        
