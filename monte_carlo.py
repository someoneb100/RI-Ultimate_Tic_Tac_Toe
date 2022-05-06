from config import NUMBER_OF_MONTE_CARLO_SIMULATIONS, CPUCT, EPS
import math
import numpy as np

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
                
        v = sum(res)/len(res)
        s = self.env.getCanonicalState()
        probs = np.zeros(81)
        for a in range(81):
            if (s,a) in self.Qsa:
                if self.Qsa[(s,a)] is not None:
                    probs[a] = self.Qsa[(s,a)]
                else:
                    probs[a] = -1
            else:
                probs[a] = 0
            
        probs+=1
        probs/=probs.sum()
        return probs, v
        
    def search(self, env):

        s = env.getCanonicalState()
        
        # ako je novootkriveno stanje pozivamo model
        if s not in self.Ps[s]:
            self.Ps[s], v = self.model.predict(env.board / 2, env.allowed_field / 8)
            return -v;
          
        #biramo sledecu akciju  
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(81):
            if (s,a) in self.Qsa:
                if self.Qsa[(s,a)] is None:
                    continue
                u = self.Qsa[(s,a)] + CPUCT * self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
            else:
                u = CPUCT * self.Ps[s][a] * math.sqrt(self.Ns.get(s,0) + EPS)
                
            if u > cur_best:
                cur_best = u
                best_act = a
               
        a = best_act
        
        #"odigraj" sledeci potez
        board, next_mini_board, reward, done, info = env.play(a)
        env.flip()
        
        #ako bi ovom akcijom usli u nevalidno stanje
        if(info.startswith('Invalid move:')):
            self.Qsa[(s,a)] = None
            self.Nsa[(s,a)] = 1
            self.Ns[s] = self.Ns.get(s,0)
            return None
        elif done: #ako bi ovom akcijom usli u zavrsno stanje
            v = -reward
        else:
            #rekurzivni poziv
            v = search(env)
        if v is not None:
        #podesi nsa ns i qsa
            if (s,a) in self.Qsa:
                self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
                self.Nsa[(s,a)] += 1
            else:
                self.Qsa[(s,a)] = v
                self.Nsa[(s,a)] = 1

            self.Ns[s] = self.Ns.get(s,0) + 1
            return -v
        
        return None
        
        
        
