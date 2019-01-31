# coding: utf-8

import io
import os
import pickle
import string
import operator
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup

def file_name_list(type):
    fileList = os.listdir('./{}'.format(type))
    if '.DS_store' in fileList:
        fileList.remove('.DS_store ')
    fileNameList = list(map(lambda x: './{}/{}'.format(type, x), fileList))
    return fileNameList

def read_html(fileName):
    html = io.open(fileName, mode="r", encoding="utf-8")
    soup = BeautifulSoup(html, 'html.parser') 
    return soup

def read_stop_words(stopFileName):
    file = open(stopFileName, mode="r", encoding="utf-8")
    words = list(map(lambda x: x[0:-1], file.readlines()))
    return words

def found_in_text(lst, value):
    lst = lst[::-1]
    found = lst.index(value) if value in lst else None
    if found:
        return found
    return None

def clean_soup(soup, stopWords):
    case = list(soup.get_text().split())
    puncs = string.punctuation.translate({ord('('): None, ord(')'): None}) + '’' + '“' + '”'
    return list(map(lambda x: x.lower(), list(filter(lambda x:  4 < len(x) <= 20 and not any(p in x for p in puncs) and x not in stopWords and x.count('(') == x.count(')'), case))))

def clean_text(lst, stopWords):
    puncs = string.punctuation.translate({ord('('): None, ord(')'): None}) + '’' + '“' + '”'
    return list(map(lambda x: x.lower(), list(filter(lambda x:  4 < len(x) <= 20 and not any(p in x for p in puncs) and x not in stopWords and x.count('(') == x.count(')'), lst))))

def case_class(fileName):
    return fileName[-fileName[::-1].index('.') - 2]


if __name__ == '__main__':

    stopFileName = './stopWords.txt'
    stopWords = read_stop_words(stopFileName)
    
    data = []
    labels = []
    fileNameListCombined = file_name_list('Combined')
    for fileName in fileNameListCombined:
        soup = read_html(fileName)
        cleanSoup = clean_soup(soup, stopWords)
        caseClass = str(case_class(fileName))
        data.append(cleanSoup)
        labels.append(caseClass)
    
    with open('textData.txt', 'w', encoding="utf-8") as file:
        for case in data:
            file.write('{}\n'.format(' '.join(case)))

    with open('textLable.txt', 'w', encoding="utf-8") as file:
        for case in labels:
            file.write('{}\n'.format(case))
    
