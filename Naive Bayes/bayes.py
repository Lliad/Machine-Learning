# author: Kitahara Kazusa
# date: 2021/2/15

from numpy import *
from os import listdir
import re
import random

def creatVocabList(dataset):
    vocabList = []
    for doc in dataset:
        vocabList.extend(set(doc))
    return list(set(vocabList))

def doc2Vec(docSet, vocabList):
    returnVec = [0] * len(vocabList)
    for i in docSet:
        if i in vocabList:
            # returnVec[vocabList.index(i)] = 1 set-of-words model
            returnVec[vocabList.index(i)] += 1 # bag-of-words model
    return returnVec

def bayes(docMat, docLabel):
    """
    :param docMat: document in vector
    :param docLabel: type of document, 1 == abuse, 0 == not aubse
    :return: possbility of abuse doc, P(each word | total word number of abuse doc),
            P(each word | total word number of non-abuse doc)
    """
    docNum = len(docLabel) # number of all doc
    wordNum = len(docMat[0]) # number of words for each doc
    pAbusive = sum(docLabel) / float(docNum)

    total0Word = 2
    total1Word = 2
    abuseVec = ones(wordNum)
    nonAbuseVec = ones(wordNum)
    for i in range(docNum):
        if docLabel[i] == 1:
            abuseVec += docMat[i]
            total1Word += sum(docMat[i])
        else:
            nonAbuseVec += docMat[i]
            total0Word += sum(docMat[i])
    abuseVec = log(abuseVec / float(total1Word))
    nonAbuseVec = log(nonAbuseVec / float(total0Word))

    return pAbusive, abuseVec, nonAbuseVec

def classifyBayes(vec, pAbusive, abuseVec, nonAbuseVec):
    p0 = sum(vec * nonAbuseVec) + math.log(1 - pAbusive)
    p1 = sum(vec * abuseVec) + math.log(pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

def docCut(longString):
    symbleList = re.split(r'\W*', str(longString))
    return [word.lower() for word in symbleList if len(word) > 2]

def test():
    docList = []
    classList = []
    filelist = listdir(r'C:\Users\86519\Desktop\ML\4.NaiveBayes\email\spam')
    for i in range(1, len(filelist) + 1):
        wordList = docCut(open(r'C:\Users\86519\Desktop\ML\4.NaiveBayes\email\spam\%d.txt' % i))
        docList.append(wordList)
        classList.append((1))
        wordList = docCut(open(r'C:\Users\86519\Desktop\ML\4.NaiveBayes\email\ham\%d.txt' % i))
        docList.append(wordList)
        classList.append(0)
    vocabList = creatVocabList(docList)

    trainingSet = range(50)
    testSet = []
    for i in range(10): # random pick 10 emails to test
        index = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[index])

    trainingMat = []
    traingingClass = []
    for index in trainingSet:
        trainingMat.append(doc2Vec(docList[index], vocabList))
        traingingClass.append(classList[index])
    pSpam, p1V, p0V = bayes(array(trainingMat), array(traingingClass))

    errCnt = 0
    for index in testSet:
        wordVec = doc2Vec(docList[index], vocabList)
        if classifyBayes(array(wordVec), pSpam, p1V, p0V) != classList[index]:
            errCnt += 1
    print('total error number is: %d' % errCnt)

if __name__ =='__main__':
    test()
