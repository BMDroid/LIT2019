# coding: utf-8

import os
import numpy as np
from collections import Counter
import operator
from sklearn.utils import shuffle


def load_dataset(fileName):
    sections = []
    labels = []
    try:
        fin = open(fileName)
    except:
        print("Error")
    for line in fin:
        t = line.split()
        sections.append(t[0:-2])
        labels.append(int(t[-2]))
    return sections, labels

def load_vocab_list(vocabFileName):
    with open(vocabFileName, 'r') as file:
        vocabList = list(map(lambda x: x[0:-1], file.readlines()))
        return vocabList

def bag_of_words_2_vec(vocabList, text):
    vec = [0] * len(vocabList)
    for word in text:
        if word in vocabList:
            vec[vocabList.index(word)] += 1
    return vec

def bag_of_words_2_vecs(vocabList, textSet):
    vecs = []
    for text in textSet:
        vecs.append(bag_of_words_2_vec(vocabList, text))
    return vecs

def auto_norm(dataSet):
    minVals = dataSet.min(0) # get the minimum values of each column and place in the miniVals
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def train_NB(trainVecs, trainLabels):
    m = len(trainVecs) # the size of the training set
    n = len(trainVecs[0]) # the length of the wordVec
    similarNum = sum(trainLabels)
    pSimilar = similarNum / m
    p0Numerator = np.ones(n)
    p0Denominator = 2
    p1Numerator = np.ones(n)
    p1Denominator = 2
    for index, vec in enumerate(trainVecs):
        if trainLabels[index] == 0:
            p0Numerator += np.array(vec)
            p0Denominator += sum(vec)
        else:
            p1Numerator += np.array(vec)
            p1Denominator += np.sum(vec)
    p0Vec = np.log(p0Numerator / p0Denominator)
    p1Vec = np.log(p1Numerator / p1Denominator)
    return p0Vec, p1Vec, pSimilar

def train_NB_test(trainVecs, trainLabels):
    m = len(trainVecs) # the size of the training set
    n = len(trainVecs[0]) # the length of the wordVec
    similarNum = sum(trainLabels)  
    pSimilar = similarNum / m
    p0Numerator = np.zeros(n)
    p0Denominator = 0
    p1Numerator = np.zeros(n)
    p1Denominator = 0
    for index, vec in enumerate(trainVecs):
        if trainLabels[index] == 0:
            p0Numerator += np.array(vec)
            p0Denominator += sum(vec)
        else:
            p1Numerator += np.array(vec)
            p1Denominator += np.sum(vec)
    p0Vec = p0Numerator / p0Denominator
    p1Vec = p1Numerator / p1Denominator
    return p0Vec, p1Vec, pSimilar
    

def classify_NB(testVec, p0Vec, p1Vec, pSimilar):
    p0 = sum(testVec * p0Vec) + np.log(1 - pSimilar)
    p1 = sum(testVec * p1Vec) + np.log(pSimilar)
    if p1 > p0:
        return 1
    return 0


def test_NB(fileName, vocabFileName, random=True, hoRatio = 0.2):
    sections, labels = load_dataset(fileName)
    vocabList = load_vocab_list(vocabFileName)
    m = len(sections)
    if random:
        sections, labels = shuffle(sections, labels)
    idx = int(m * (1 - hoRatio)) + 1
    trainSections = sections[0: idx]
    trainLabels = labels[0:idx]
    testSections = sections[idx::]
    testLabels = labels[idx::]
    trainVec = bag_of_words_2_vecs(vocabList, trainSections)
    p0Vec, p1Vec, pSimilar = train_NB(trainVec, trainLabels)   
    for i, test in enumerate(testSections):
        testVec = np.array(bag_of_words_2_vec(vocabList, test))
        print("{} Section is classified as: {}. And it's original label is {}".format(i, classify_NB(testVec, p0Vec, p1Vec, pSimilar), testLabels[i]))

if __name__ == '__main__':
    fileName = 'Likelyhood of Confusion Data.txt'
    vocabFileName = 'Likelyhood of Confusion Vocab copy.txt'
    test_NB(fileName, vocabFileName)


