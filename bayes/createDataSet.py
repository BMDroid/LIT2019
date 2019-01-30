import os
import csv
import string
import numpy as np
import operator
from collections import Counter

similarityDict = {'Marks Similarity': 0, 'Visual Similarity':1, 'Aural Similarity':2, 'Conceptual Similarity': 3, 'G&S Similarity': 4, 'Likelyhood of Confusion': 5}

def load_data(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        lst = list(reader)
    data = []
    labels = []
    conclusions = []
    for l in lst: # if the table is complete, the slicing will not be needed
        case = []
        label = list(map(int, l[8].strip().replace('X', '-1').split('_')))
        conclusion = 1 if l[4] == 'OS' else 0
        case.extend(l[9:15])
        data.append(case)
        labels.append(label)
        conclusions.append(conclusion)
        print(conclusions)
    return dataSet, labels, conclusions

def read_stop_words(stopFileName):
    file = open(stopFileName)
    stopWords = list(map(lambda x: x[0:-1], file.readlines()))
    return stopWords

def clean_text(lst, stopWords):
    puncs = string.punctuation + '’' + '“' + '”'
    return list(map(lambda x: x.lower(), list(filter(lambda x:  4 < len(x) <= 20 and not any(p in x for p in puncs) and x not in stopWords, lst)))) 

def create_section_data(dataSet, labels, conclusions, sectionIdx, stopWords):
    lst = []
    for i, case in enumerate(dataSet):
        content = case[sectionIdx].split()
        label = str(labels[i][sectionIdx])
        conclusion = str(conclusions[i])
        if label != -1:
            cleanContent = clean_text(content, stopWords)
            cleanContent.append(label)
            cleanContent.append(conclusion)
            lst.append(cleanContent)
        else:
            continue
    return lst

def create_vocab_list(dataSet, sectionIdx, stopWords, size):
    vocabList = []
    sectionIdx = similarityDict[section]
    for case in dataSet:
        caseText = case[sectionIdx].split()
        vocabList.extend(caseText)
    cleanVocabList = clean_text(vocabList, stopWords)
    vocabDict = dict(Counter(cleanVocabList))
    vocabDictSorted = sorted(vocabDict.items(), key=operator.itemgetter(1), reverse=True)
    return list(map(lambda x: x[0], vocabDictSorted))[0:size]

if __name__ == '__main__':
    fileName = './Data.csv'
    stopFileName = './stopWordsNot.txt'
    stopWords = read_stop_words(stopFileName)

    dataSet, labels, conclusions = load_data(fileName)
    
    for section, sectionIdx in similarityDict.items():
        sectionData = create_section_data(dataSet, labels, conclusions, sectionIdx, stopWords)
        with open(f'{section}Data.txt', 'w') as file:
            file.writelines(' '.join(i) + '\n' for i in sectionData)
        size = 50
        sectionVocab = create_vocab_list(dataSet, sectionIdx, stopWords, size)
        with open(f'{./section}Vocab.txt', 'w') as f:
            for word in sectionVocab:
                f.write("%s\n" % word)

    








