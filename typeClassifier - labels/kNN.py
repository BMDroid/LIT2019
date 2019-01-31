import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator


def load_data(fileName):
    data = np.loadtxt(fileName)
    m, n = np.shape(data)
    X = data[:, 0: n - 1]
    Y = data[:, -1]
    return X, Y, m, n

def auto_norm(dataSet):
    minVals = dataSet.min(0) # get the minimum values of each column and place in the miniVals
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet # create an array that has the same dimension of the dataSet containning inX
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() # return the indices of the sorted array from the smallest to the largest
    classCount = {}
    for i in range(k):
      voteIlabel = labels[sortedDistIndicies[i]]
      classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # return the label of the item occurring the most frequently.


def class_test(fileName, k, random=True, hoRatio=0.1):
    # hoRatio: the ratio of the test set of the whole data set
    dataMat, labels, _, _ = load_data(fileName)
    if random:
        np.random.shuffle(dataMat)
    normMat, _, _ = auto_norm(dataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio) # the size of the test set
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify(normMat[i,:], normMat[numTestVecs:m,:], labels[numTestVecs:m], k)
        # print(f"The classifier came back with: {classfierResult}, the real answer is: {labels[i]}")
        if(classfierResult != labels[i]): errorCount += 1.0
    print(f"{k} The total error rate is: {errorCount / float(numTestVecs)}")


if __name__ == '__main__':
    numLables = 3
    wordsBagSize = 1365
    fileName = f'./{numLables}dataCombined{wordsBagSize}.txt'
    random = True
    hoRatio = 0.1
    for k in range(1, 10):
        class_test(fileName, k, random, hoRatio)
    