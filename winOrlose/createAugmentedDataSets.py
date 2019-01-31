# coding: utf-8

import os
import csv
import string
import numpy as np
import operator
import random
import copy 
from collections import Counter

def load_data(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        lst = list(reader)
    dataSet = []
    categories = []
    for l in lst: # if the table is complete, the slicing will not be needed
        label = 1 if l[0] == 'OS' else 0
        data = list(map(int, l[1].replace('X', '-1').split('_')))
        category= l[2]
        data.append(label)
        dataSet.append(data)
        categories.append(category)
    return dataSet, categories

def augment_data(dataSet):
    augmentedData = copy.deepcopy(dataSet)
    for data in augmentedData:
        idx = random.randint(0, 5) # random index for the data
        augmented = random.random()
        data[idx] += augmented
    return augmentedData


if __name__ == '__main__':
    fileName = '/Users/bmdroid/workspace/github/lit2019/winOrlose//winOrLoseTest.csv'
    dataSet, categories = load_data(fileName)
    # create another n sets of augemented data
    n = 10
    augmentDataSet = []
    for _ in range(n):
        augmentedDataSet = augment_data(dataSet)
        augmentDataSet.extend(augmentedDataSet)
    dataSet.extend(augmentDataSet)
    with open('data.txt', 'w') as file:
        file.writelines(' '.join(map(str, data)) + '\n' for data in dataSet)
    
    # save the data as csv file
    dataSet.insert(0, ['MarksSimilarity', 'VisualSimilarity', 'AuralSimilarity', 'ConceptualSimilarity', 'G&SSimilarity', 'LikelyhoodOfConfusion', 'WinOrLose'])
    np.savetxt("data.csv", dataSet, delimiter=",", fmt='%s')
    '''
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataSet)
    '''
    
