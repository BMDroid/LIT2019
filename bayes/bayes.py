from numpy import *


def load_dataset(fileName):
    ''' Load the dataset from the text file. And return its feature matrix and class labels in lists.
    Args:
        fileName::str
            The name of the file storing the dataset.
    Returns:
        (postingList, classList)::([str], [int])
            postingList::[str]
                The posts stored in a list, and each element is the all tokens from the posts in a list.
            classList::[int]
                The class label for each posts stored in a list.
                0: the post is not abusive.
                1: the post is abusive.
    '''
    postingList = []
    classList = []
    try:
        fin = open(fileName)
    except:
        print(f"Something wrong when openning {fileName}.")
    for line in fin:
        t = line.split()
        postingList.append(t[0:-1])
        classList.append(int(t[-1]))
    return postingList, classList


def create_vocab_list(dataSet):
    ''' Create the vocabulary list from all the posts. And store them in a sorted list.
    Args:
        dataSet::[[str]]
            The nested list, each element is the splited post in a list.
    Returns:
        vocabSet::[str]
            The list of strings consists of all the vocabulary from all the posts.
    '''
    vocabSet = set()
    for document in dataSet:
        # | operator is used for union of two sets, same as union()
        # vocabSet = vocabSet.union(set(document))
        vocabSet = vocabSet | set(document)
    return sorted(vocabSet)


# a set of words can only have one occurrence of each word
def set_of_words_2_vec1(vocabList, inputText):
    ''' Create the token vector of the input text using the vocabulary list.
    Args:
        vocabList::[str]
            The list of tokens contains all the vocabulary from all the posts.
        inputText::str
            The input text in a string.
    Returns:
        vec::[int]
            A token vector.
            0: the input text has no occurence of the token
            1: the input text has token inside.
    '''
    vec = [0] * len(vocabList)
    for word in inputText:
        if word in vocabList:
            vec[vocabList.index(word)] = 1
        else:
            print(f"\'{word}\' is not in my vocabulary.")
    return vec


def set_of_words_2_vec2(vocabList, inputSet):
    ''' Create the token vectors from a list of inputs.
    Args:
        vocabList::[str]
            The list of tokens contains all the vocabulary from all the posts.
        inputSet::[str]
            Input texts stored in a list of strings.
            Each element is one input text string.
    Returns:
        vecList::[[int]]
            Token vectors for each input text from the input set. Each element is one token vector.
            0: the input text has no occurence of the token
            1: the input text has token inside.
    '''
    vecList = []
    for inputText in inputSet:
        vecList.append(set_of_words_2_vec1(vocabList, inputText))
    return vecList


def set_of_words_2_vec3(vocabList, inputText):
    ''' Create the token vector of the input text using the vocabulary list.
    Only when the length of the inputText is bigger than the size of the vocabList.
    Args:
        vocabList::[str]
            The list of tokens contains all the vocabulary from all the posts.
        inputText::str
            The input text in a string.
    Returns:
        vec::[int]
            A token vector.
            0: the input text has no occurence of the token
            1: the input text has token inside.
    '''
    vec = [0] * len(vocabList)
    for index, word in enumerate(vocabList):
        if word in inputText:
            vec[index] = 1
        else:
            print(f"\'{word}\' is not in my vocabulary.")
    return vec


# 4.4 Naive Bayes bag-of-words model
# a bag of words can have multiple occurrences of each word
def bag_of_words_2_vec(vocabList, inputText):
    ''' Create the token vector of the input text using the vocabulary list.
    Args:
        vocabList::[str]
            The list of tokens contains all the vocabulary from all the posts.
        inputText::str
            The input text in a string.
    Returns:
        vec::[int]
            A token vector.
            The integer represents the time of occurrence of the token in the input text.
    '''
    vec = [0] * len(vocabList)
    for word in inputText:
        if word in vocabList:
            vec[vocabList.index(word)] += 1
        else:
            print(f"\'{word}\' is not in my vocabulary.")
    return vec


def train_NB0(trainMatrix, trainCategory):
    ''' Train the naive Bayes classifier. And return the known values for calculate the conditioal probability.
    Args:
        trainMatrix::[[int]]
            The list of token vectors created from all the input text in the data set.
        trainCategory::[int]
            The list of class labels for all the input in the data set.
    Returns:
        p0Vec::[float]
            The frequency/ probability of tokens of all the words in class 0.
        p1Vec::[float]
            The frequency/ probability of tokens of all the words in class 1.
        pAbusive::float
            The frequency/ probability of class 1 in the whole dataset.
    '''
    trainingSize = len(trainMatrix) # m
    vecSize = len(trainMatrix[0]) # n
    abusiveNum = sum(trainCategory) # abusive 1 and non-abusive zero
    pAbusive = abusiveNum / trainingSize
    p0Numerator = zeros(vecSize)
    p0Denominator = 0
    p1Numerator = zeros(vecSize)
    p1Denominator = 0
    for index, example in enumerate(trainMatrix):
        if trainCategory[index] == 0:
            # create an array from a list
            p0Numerator += array(example)
            p0Denominator += sum(example)
        else:
            p1Numerator += array(example)
            p1Denominator += sum(example)
    p0Vec = p0Numerator / p0Denominator
    p1Vec = p1Numerator / p1Denominator
    return p0Vec, p1Vec, pAbusive


def train_NB1(trainMatrix, trainCategory):
    ''' Train the naive Bayes classifier. And return values for calculate the conditioal probability.
        Modify the classifier for the real-word calculation.
        Natural logarithm for conditional probabilities.
        Initialize the numerator to 1 for all the tokens.
        Initialize the denominator to 2.
    Args:
        trainMatrix::[[int]]
            The list of token vectors created from all the input text in the data set.
        trainCategory::[int]
            The list of class labels for all the input in the data set.
    Returns:
        p0Vec::[float]
            The frequency/ probability of the tokens in class 0.
        p1Vec::[float]
            The frequency/ probability of the tokens in class 1.
        pAbusive::float
            The frequency/ probability of class 1/ abusive in the whole dataset.
    '''
    trainingSize = len(trainMatrix)
    vecSize = len(trainMatrix[0])
    abusiveNum = sum(trainCategory)
    pAbusive = abusiveNum / trainingSize
    # initialize all the occurence in numerator to 1
    p0Numerator = ones(vecSize)
    # initialize the denominator to 2
    p0Denominator = 2
    p1Numerator = ones(vecSize)
    p1Denominator = 2
    for index, example in enumerate(trainMatrix):
        if trainCategory[index] == 0:
            # create an array from a list
            p0Numerator += array(example)
            p0Denominator += sum(example)
        else:
            p1Numerator += array(example)
            p1Denominator += sum(example)
    # use the natural logarithm
    # to aviod the underflow and the round-off problem in Python causing by many small values product
    p0Vec = log(p0Numerator / p0Denominator)
    p1Vec = log(p1Numerator / p1Denominator)
    return p0Vec, p1Vec, pAbusive


# 4.3 Naive Bayes classify function
def classify_NB(vec2Classify, p0Vec, p1Vec, pClass1):
    ''' Naive Bayes Clafssifier.
    Args:
        vec2Classify::[int]
            The token vector of the input text waiting for classifying.
        p0Vec::[float]
            The frequency/ probability of the tokens in class 0.
        p1Vec::[float]
            The frequency/ probability of the tokens in class 1.
        pClass1::float
            The frequency/ probability of class 1 in the whole dataset.
    Returns:
        class:int
            The class label of the prediction.
    '''
    # here the multiplication is element wise
    # and the p(w) was set to 1
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    if p1 > p0:
        return 1
    return 0


def test_NB():
    ''' Test the naive Bayes classifier.
    '''
    fileName = 'posting.txt'
    listOfPosts, listClasses = load_dataset(fileName)
    myVocabList = create_vocab_list(listOfPosts)
    trainMatrix = set_of_words_2_vec2(myVocabList, listOfPosts)
    p0Vec, p1Vec, pAbusive = train_NB1(trainMatrix, listClasses)
    inputText1 = 'love my dalmation'.split()
    inputText1Vec = array(set_of_words_2_vec1(myVocabList, inputText1))
    print(f"{inputText1} is classified as: {classify_NB(inputText1Vec, p0Vec, p1Vec, pAbusive)}.")
    inputText2 = 'stupid garbage'.split()
    inputText2Vec = array(set_of_words_2_vec1(myVocabList, inputText2))
    print(f"{inputText2}' is classified as: {classify_NB(inputText2Vec, p0Vec, p1Vec, pAbusive)}.\n")


# 4.5 File parsing and full spam test functions
def text_parse(s):
    ''' Take a string and parses out the text into a list of strings.
    It eliminates anything under two char and conters them to lower case.
    Args:
        s::str
            The input string.
    Returns:
        t::[str]
            A list of strings parsing from the input string.
    '''
    import string
    t = []
    # table = str.maketrans(dict(zip(string.punctuation, [' ']*len(string.punctuation))))
    # dictionary.fromkeys(sequence[, value])
    # value is optional, here the value = ' '
    # https://www.programiz.com/python-programming/methods/dictionary/fromkeys
    table = str.maketrans(dict.fromkeys(string.punctuation, ' '))
    s = s.lower().translate(table)
    t = s.split()
    t = [word for word in t if len(word) > 2]
    return t


def spam_test():
    ''' A simple test for the naive bayes spam email classifier.
    The test set is called hold-out cross validation.
    '''
    import random
    docList = []
    classList = []
    fullText = []  # useless
    # build the dataset
    for i in range(1, 26):
        # file.read() return the file content in a string
        wordList = text_parse(
            open('email/spam/{:d}.txt'.format(i), errors='ignore').read())
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)
        wordList = text_parse(
            open('email/ham/{:d}.txt'.format(i), errors='ignore').read())
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)
    myVocabList = create_vocab_list(docList)
    # split the dataset to 40 training examples and 10 test examples
    # use + operator instead of using append(), because append() not return a new list
    dataSet = [docList[i] + [classList[i]]
               for i in range(len(docList))]
    random.shuffle(dataSet)
    testDataSet = dataSet[:10]
    trainingDataSet = dataSet[10:]
    testSet = [example[:-1] for example in testDataSet]
    testClasses = [example[-1] for example in testDataSet]
    trainingSet = [example[:-1] for example in trainingDataSet]
    trainingClasses = [example[-1] for example in trainingDataSet]
    # train the naive bayes using the training set
    trainingMatrix = set_of_words_2_vec2(myVocabList, trainingSet)
    p0Vec, p1Vec, pSpam = train_NB1(trainingMatrix, trainingClasses)
    # test the classifier on the test set and calculate the error rate
    errorCount = 0
    testMatrix = set_of_words_2_vec2(myVocabList, testSet)
    for i, testVec in enumerate(testMatrix):
        prediction = classify_NB(testVec, p0Vec, p1Vec, pSpam)
        if prediction != testClasses[i]:
            errorCount += 1
            print(f"Classification error happens: {testSet[i]}")
    print(f"The error rate is {errorCount / len(testSet)}")


# 4.6 Rss feed classifier andf frequent word removal functions
def calc_most_freq(vocabList, fullText):
    ''' Calculate the 30 most frequent words in the vocabulary list,
        which are commonly classified as the stop words.
    Args:
        vocabList::[str]
            The list of tokens contains all the vocabulary from all the posts.
        fullText::[str]
            The list of words consists of all the posts.
    Returns:
        sortedDict[:30]::[(token, occurrence)]
            token::str
                The token.
            occurence::int
                The occurrence of the token.
    '''
    import operator
    vocabDict = {}
    for word in vocabList:
        vocabDict[word] = fullText.count(word)
    sortedDict = sorted(vocabDict.items(),
                        key=operator.itemgetter(1), reverse=True)
    return sortedDict[:30]


if __name__ == '__main__':
    fileName = 'posting.txt'
    listOfPosts, listClasses = load_dataset(fileName)
    print(listOfPosts)
    # print() has a default parameter 'end', and its value are '\n'.
    print(listClasses, "\n")

    myVocabList = create_vocab_list(listOfPosts)
    # override the defalut value to print an empty line
    print(myVocabList, end="\n\n")

    vec0 = set_of_words_2_vec1(myVocabList, listOfPosts[0])
    print(vec0, "\n")

    vecSet = set_of_words_2_vec2(myVocabList, listOfPosts)

    p0Vec, p1Vec, pAusive = train_NB0(vecSet, listClasses)
    print(pAusive)
    print(p1Vec)
    # argmax(axis=a) return the index of the maximum element in axis a
    print(myVocabList[p1Vec.argmax()], "\n")

    test_NB()

    s = 'this is a test string contains The www.google.com/new_search?ml'
    print(text_parse(s), "\n")

    spam_test()
