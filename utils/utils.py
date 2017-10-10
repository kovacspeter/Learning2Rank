import itertools
import numpy as np

def pairwise_transform(X, y):
    # form all pairwise combinations
    comb = itertools.combinations(range(X.shape[0]), 2)
    k = 0
    Xp, yp= [], []
    for (i, j) in comb:
        if y[i] == y[j]:
            # skip if same target
            continue
        Xp.append(X[i] - X[j])
        diff = y[i] - y[j]
        yp.append(np.sign(diff))
        # output balanced classes
        if yp[-1] != (-1) ** k:
            yp[-1] *= -1
            Xp[-1] *= -1
            diff *= -1
        k += 1

    return Xp, yp


