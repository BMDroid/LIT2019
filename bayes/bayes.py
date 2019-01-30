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
    # return list(map(lambda x: x / len(text), vec))
    return vec

def bag_of_words_2_vecs(vocabList, textSet):
    vecs = []
    for text in textSet:
        vecs.append(bag_of_words_2_vec(vocabList, text))
    return vecs

def train_naive_bayes_multi(trainVecs, trainLabels):
    m = len(trainVecs) # the size of the training set
    n = len(trainVecs[0]) # the length of the wordVec
    labels = set(trainLabels)
    labelDict = dict(Counter(trainLabels))
    pDict = {}
    for l in labels:
        p = labelDict[l] / m
        pNumerator = np.zeros(n)
        pDenominator = 0
        for index, vec in enumerate(trainVecs):
            if trainLabels[index] == l:
                pNumerator += np.array(vec)
                pDenominator += np.sum(vec)
        pVec = pNumerator / pDenominator
        pDict[l] = [pVec, p]
    return pDict


def train_NB_multi(trainVecs, trainLabels):
    m = len(trainVecs) # the size of the training set
    n = len(trainVecs[0]) # the length of the wordVec
    labels = set(trainLabels)
    labelDict =  dict(Counter(trainLabels))
    pDict = {}
    for l in labels:
        p = labelDict[l] / m
        pNumerator = np.ones(n)
        pDenominator = 2
        for index, vec in enumerate(trainVecs):
            if trainLabels[index] == l:
                pNumerator += np.array(vec)
                pDenominator += np.sum(vec)
        pVec = np.log(pNumerator / pDenominator)
        pDict[l] = [pVec, p]
    return pDict


def classify_NB(vec2Classify, pDict):
    pBayes = {}
    for l in pDict:
        p = sum(vec2Classify * pDict[l][0]) + np.log(pDict[l][1])
        pBayes[l] = p
    label = max(pBayes.items(), key=operator.itemgetter(1))[0]
    return label


def test_NB(fileName, vocabFileName, random=True, hoRatio = 0.1):
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
    pDict = train_NB_multi(trainVec, trainLabels)

    # errorCount = 0
    for i, test in enumerate(testSections):
        testVec = np.array(bag_of_words_2_vec(vocabList, test))
        print("{} Section is classified as: {}. And it's original label is {}".format(i, classify_NB(testVec, pDict), testLabels[i]))

if __name__ == '__main__':
    fileName = 'Visual Similarity Data.txt'
    vocabFileName = 'Visual Similarity Vocab.txt'
    test_NB(fileName, vocabFileName)


