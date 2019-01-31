import os
import pickle
import numpy as np
import pandas
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble


def load_data(fileName):
    dataSet = []
    with open(fileName, 'r', encoding='utf-8') as file:
        dataSet = [(line.strip())[:-1] for line in file.readlines()]
    return dataSet

def load_label(fileName):
    labels = []
    with open(fileName, 'r', encoding='utf-8') as file:
        labels = [int(line) for line in file.readlines()]
    return labels

if __name__ == '__main__':
    fileName = 'textData.txt'
    dataSet = load_data(fileName)

    fileName = 'textLable.txt'
    labels = load_label(fileName)

    '''
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(dataSet)
    counts = X.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(counts)

    vectorizer = TfidfVectorizer()
    vec = vectorizer.fit_transform(dataSet)
    print(type(vec))
    print(vec.shape)
    '''

    trainDF = pandas.DataFrame()
    trainDF['text'] = dataSet
    trainDF['label'] = labels


    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)