import time

from sklearn.linear_model import LogisticRegression
import numpy as np

from utils.data_loaders import load_mslr
from utils.eval_metrics import ndcg
from utils.utils import pairwise_transform

start_time = time.time()
data = load_mslr('/home/pkovacs/Documents/data/MSLR-WEB10K/Fold1')
print("---Data reading took %s seconds ---" % (time.time() - start_time))

X_train, y_train = [], []
X_test, y_test = {}, {}
i = 0
for key in data.keys():
    i += 1
    if np.random.random() < 0.8:
        X, y = pairwise_transform(np.array(data[key]['X']), np.array(data[key]['y']))
        X_train.extend(X)
        y_train.extend(y)
    else:
        X_test[key] = data[key]['X']
        y_test[key] = data[key]['y']

X_train, y_train = np.array(X_train), np.array(y_train)

# Pair-wise ranking with logisic regression
start_time = time.time()
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("---Logistic Regression training took %s seconds ---" % (time.time() - start_time))
ndcgs = []
for key in X_test:
    y = np.array(y_test[key])
    X = np.array(X_test[key])
    # print(X_test[key], type(X_test[key]))
    # print(y_test[key], type(y_test[key]))

    predicted = np.max(lr.predict_proba(X), 1)
    predicted = np.sort(predicted)[::-1]
    ndcg_val = ndcg(y, predicted, 10)
    ndcgs.append(ndcg_val)
    print('Logistic Regression NDCG@10 = ', ndcg_val)

print(np.mean(ndcgs))
