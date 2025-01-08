import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    y = data["class"]
    X = data.drop(['class'], axis=1)
    y = y.to_numpy()
    X = X.to_numpy()
    return X, y

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)
