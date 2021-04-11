import numpy as np
from scipy.stats import bernoulli


classifiers = [0.6]*10 + [0.75]

def get_pred(classifiers):
    preds = [bernoulli.rvs(p, size=1)[0] for p in classifiers]
    m = np.floor((len(classifiers)/2)+1)
    pred = 1 if sum(preds) >= m else 0
    return pred

preds = [get_pred(classifiers) for x in range(1,10000)]
print(sum(preds)/len(preds))
