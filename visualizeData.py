import os
import pickle
import string
import operator
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup

def load_data(fileName):
    data = np.loadtxt(fileName)
    m, n = np.shape(data)
    X = data[:, 0: n - 4]
    Y = data[-3::]
    return X, Y, m, n

def plot_hist(vec, bins=20):
    plt.hist(vec, bins)

if __name__ == '__main__':
    fileName = 'datacombined.txt'
    X, Y, m, n = load_data(fileName)
    plt.hist(X[1], bins=20)
    plt.show()
