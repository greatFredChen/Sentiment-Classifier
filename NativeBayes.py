import numpy as np


def train_bayes(x, y):
    m = len(x)
    features = x.shape[1]
    num_classes = []
    y_unique = np.unique(y)
    for y_i in y_unique:
        num_yi = np.sum([1 if y_i == y[i] else 0 for i in range(m)])
        num_classes.append(num_yi)
    pClass = np.array(num_classes) / m
    pNumClass = np.array([np.ones(features) for i in y_unique])
    pNumDenom = [2.0 for i in y_unique]

    for i in range(m):
        for j in range(len(y_unique)):
            if y[i] == y_unique[j]:
                pass
