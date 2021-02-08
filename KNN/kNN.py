# author: Kitahara Kazusa
# date: 2021/2/6

from numpy import *
import operator
from os import listdir
from collections import Counter

training_file_path = r"C:\Users\86519\Desktop\ML\2.KNN\trainingDigits"
test_file_path = r"C:\Users\86519\Desktop\ML\2.KNN\testDigits"
K = 2


def classifyKNN(index, dataSet, lable, k):
    """
    :param index: vector waiting to be classified
    :param dataSet: training dataset
    :param lable: training dataset'answer
    :param k: kNN, k-nearest nabor

    a.shape: return the dimension of a; eg: if a is a 2*3 matrix, a.shape = (2, 3)
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(index, (dataSetSize, 1)) - dataSet
    diffMat = diffMat ** 2
    distance = (diffMat.sum(axis=1)) ** 0.5
    sortDistId = distance.argsort()

    classCnt = {}
    for i in range(k):
        possLable = lable[sortDistId[i]]
        classCnt[possLable] = classCnt.get(possLable, 0) + 1
    maxPoss = max(classCnt)
    return maxPoss


"""
image2vector: 32*32 vector -------> 1*1024 vector
"""


def image2vector(filename):
    vec = zeros((1, 1024))
    fp = open(filename)
    for i in range(32):
        lineStr = fp.readline()
        for j in range(32):
            vec[0, 32 * i + j] = int(lineStr[j])
    return vec


def handWritingClassify():
    hwLabels = []
    trainingFileList = listdir(training_file_path)
    fileNum = len(trainingFileList)
    trainingMat = zeros((fileNum, 1024))

    for i in range(fileNum):
        fileName = trainingFileList[i]
        fileStr = fileName.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i, :] = image2vector(r'C:\Users\86519\Desktop\ML\2.KNN\trainingDigits\%s'
                                         % fileName)

    testFilelist = listdir(test_file_path)
    errCnt = 0.0
    testNum = len(testFilelist)

    for i in range(testNum):
        fileName = testFilelist[i]
        fileStr = fileName.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        testVec = image2vector(r'C:\Users\86519\Desktop\ML\2.KNN\testDigits\%s'
                               % fileName)
        classifyRes = classifyKNN(testVec, trainingMat, hwLabels, K)
        if (classifyRes != classNum):
            errCnt += 1.0
    print('The total number of error is: %d' % errCnt)
    print('The error rate is: %f' % float(errCnt / testNum))


if __name__ == '__main__':
    handWritingClassify()
