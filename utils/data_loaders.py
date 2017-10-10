import numpy as np
import os

def load_mslr(data_path):
    """
    Learning to rank datasets from Microsoft Research:
        https://www.microsoft.com/en-us/research/project/mslr/
    """
    data = {}
    for file in ['train.txt', 'test.txt', 'vali.txt']:
        with open(os.path.join(data_path, file), 'r') as f:
            for line in f:
                splitted = line.split(' ')
                key = splitted[1].split(":")[1]

                # splitted[2:-1] = [relevancy, qid, 'feature_1:value', ..., 'feature_n:value', '\n']
                X = [float(feature.split(":")[1]) for feature in splitted[2:-1]]
                y = [int(splitted[0])]

                if key not in data:
                    data[key] = {'X': [], 'y': []}
                data[key]['X'].append(np.array(X))
                data[key]['y'].append(y)

    return data
