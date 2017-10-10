import numpy as np

# Evaluation function of NDCG@10
def ndcg(y_true, y_score, k=10):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    y_true_sorted = sorted(y_true, reverse=True)
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
    dcg = 0
    argsort_indices = np.argsort(y_score)[::-1]
    for i in range(k):
        dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
    ndcg = dcg / ideal_dcg
    return ndcg

# Mean average precession MAP@10
def map(y_true, y_score, k=10):
    pass


