import numpy as np
from scipy.stats import bernoulli
from scipy import stats
from matplotlib import pyplot as plt

def get_sim_score(w):
    classifiers = 10*[.6] + [0.75]
    weights = [0.1]*10 + [w]
    
    def majority_vote (preds, weights):
        #sum of the weights
        s = sum(weights)
         #score needed for a majority vote:
        m = s/2
        preds = [(x[0]*x[1]) for x in zip(preds, weights)]
        
        pred = 1 if sum(preds) > m else 0
        
        return pred
        
    
    def get_pred(classifiers):
        preds = [bernoulli.rvs(p, size=1)[0] for p in classifiers]
        pred = majority_vote(preds, weights)
        return pred
    
    preds = [get_pred(classifiers) for x in range(1,10000)]
    return sum(preds)/len(preds)
    #print(sum(preds)/len(preds))

w =  np.linspace(0.1,0.9,9)

probs = []
for i in w:
    probs.append(get_sim_score(i))

plt.plot(w,probs)
plt.show() 
    
    
