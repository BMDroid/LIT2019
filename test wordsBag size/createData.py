import os
import pickle
import string
import operator
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup

def file_name_list(type):
    fileList = os.listdir(f'./{type}')
    if '.DS_store ' in fileList:
        fileList.remove('.DS_store ')
    fileNameList = list(map(lambda x: f'./{type}//' + x, fileList))
    return fileNameList

def read_html(fileName):
    html = open(fileName)
    soup = BeautifulSoup(html, 'html.parser') 
    return soup

def read_stop_words(stopFileName):
    file = open(stopFileName)
    words = list(map(lambda x: x[0:-1], file.readlines()))
    return words

def class_contents(soup, className):
    lst = soup.find_all(class_=className)
    text = ''.join([''.join(l.findAll(text=True)) + ' ' for l in lst])
    return text

def list_classes(soup):
    classes = [value for element in soup.find_all(class_=True) for value in element["class"]]
    return classes

def found_in_class(soup, className, value):
    lst = soup.find_all(class_=className)
    return any(value in l.findAll(text=True) for l in lst)

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

def get_headings(classList):
    lst = ['Judg-Heading-']
    headingsList = set(filter(lambda x: x if lst[0] in x else None, classList))
    return headingsList

def has_conclusion(soup, headingsList):
    for heading in headingsList:
        if found_in_class(soup, heading, 'Conclusion'):
            return True
    return False

def case_class(fileName):
    return fileName[-fileName[::-1].index('.') - 2]

def tf_idf_vectorizer(cleanSoup, vecLength, wordsBag):
    length = len(cleanSoup)
    vec = np.zeros(vecLength + 3) # for adding other labels
    for idx, word in enumerate(wordsBag):
        vec[idx] = 1000 * cleanSoup.count(word) / length
    return vec

def words_bag(type, dic):
    wordsBag = []
    fileNameList = file_name_list(type)
    for fileName in fileNameList:
        soup = read_html(fileName)
        wordsBag.extend(clean_soup(soup, stopWords))
    wordsBagHist = dict(Counter(wordsBag))
    sortedBag = sorted(wordsBagHist.items(), key=operator.itemgetter(1), reverse=True)
    mostFreqWords = list(sortedBag[0: dic[type]])
    wordsBag = list(map(lambda x: x[0], mostFreqWords))
    return wordsBag

def words_bag_combined(dic):
    wordsBag = []
    for type in dic:
        wordsBag.extend(words_bag(type, dic))
    wordsBag = list(set(wordsBag))
    return len(wordsBag), wordsBag


if __name__ == '__main__':

    stopFileName = 'stopWords.txt'
    stopWords = read_stop_words(stopFileName)

    designedBagSize = 1500
    dic = {'1': int(designedBagSize * 0.3), '2':int(designedBagSize * 0.05), '3': int(designedBagSize * 0.65)}
    
    vecLength, wordsBagCombined = words_bag_combined(dic)
    print(vecLength)
    dic['Combined']= vecLength

    wordsBagName = f'wordsBagCombined{vecLength}.pkl'
    with open(wordsBagName, 'wb') as f:
        pickle.dump(wordsBagCombined, f)
    with open(wordsBagName, 'rb') as f:
        wordsBag = pickle.load(f)
    
    dataList = []
    type = 'Combined'
    fileNameListCombined = file_name_list(type)
    for fileName in fileNameListCombined:
        soup = read_html(fileName)
        classList = list_classes(soup)
        headingList = get_headings(classList)
        hasConclusion = has_conclusion(soup, headingList)
        winOrLose = 1 if 'OS' in fileName else 0
        caseClass = case_class(fileName)
        cleanSoup = clean_soup(soup, stopWords)
        vec = tf_idf_vectorizer(cleanSoup, dic[type], wordsBag)
        vec[-3] = int(hasConclusion)
        vec[-2] = winOrLose
        vec[-1] = caseClass
        dataList.append(vec)
    dataName = f'data{type}{vecLength}.txt'
    np.savetxt(dataName, tuple(dataList))



