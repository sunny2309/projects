from math import log
import operator
import random
from random import randint
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import copy
import json
#[250,10], [1000, 25], for C
# [2500, 10], [2500 25] for I
label_map = {'e':'Edible','p':'Poison'}
cnt = 0
classLabel = None
##########INPUT
def get_non_negative_int(prompt):
    while True:
        prompt = int(input("Please enter training set size (must be a multiple of 250! and also <= 1000:)"))
        try:
            prompt = int(prompt)
            #if int(prompt) > 249 and int(prompt) < 1001 and int(prompt) % 250 == 0:
        except ValueError:
            print("Your input cannot be processed. Please enter a correct number")
            continue
        if prompt < 0:
            print("Input cannot be negative")
            continue
        else:
            break
    return prompt

def get_non_out_int(prompt2):
    while int(prompt2) ==10 or int(prompt2) == 25 or int(prompt2) == 50:
        try:
            prompt2 = int(input(prompt2))
        except ValueError:
            print("Your input needs to be one of the values. Please enter a correct number")
            continue
        if prompt2 < 0:
            print("Input cannot be negative")
            continue
        else:
            break
    return prompt2
    
#################
def createDataSet1(fileName, prompt, prompt2):
    trainingData = []
    testData = []
    labels = [      'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                    'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population',
                    'habitat']
    with open(fileName) as txtData:
        lines = np.array(txtData.readlines())
        print('Full Dataset Size : ', len(lines))
        examplesLines = list(range(len(lines)))
        TrainingSetLines,TestSetLines = [],[]
        TrainingIdx = []
        TestIndex = []
        #rnd = random.Random(123)
        TrainingIdx = np.array(random.sample(range(len(lines)), prompt))
        TestIdx = np.array([i for i in np.arange(len(lines)) if i not in TrainingIdx])
        TrainingSetLines = lines[TrainingIdx].tolist()
        TestSetLines = lines[TestIdx].tolist()

        for myline in TrainingSetLines:
            lineData = myline.strip().split(' ')
            trainingData.append(lineData)
            
        for myline in TestSetLines:
            lineData = myline.strip().split(' ')
            testData.append(lineData)
            
    return trainingData, testData, labels

def splitData(dataSet):
        character = []
        label = []
        for i in range(len(dataSet)):
            character.append([str(tk) for tk in dataSet[i][:-1]])
            label.append(dataSet[i][-1])
        return np.array(character), np.array(label)
        
#########START THE TREE
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  #
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
    
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature
    
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
def createTree(dataSet, labels, featLabels):
    #print(dataSet, labels, featLabels)
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:
        #It'll go here when there is only last feature is remaining
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree
    
def classify(inputTree, featLabels, testVec):
    global classLabel
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
    
def classifyAll(inputTree, featLabels, testDataSet):
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll
    
def combineList(testData):
    testLabel = []
    for i in range(len(testData)):
        testLabel.append(testData[i][-1])
    return testLabel
    
def countLabel(classLabelAll, testLabel):
    count = 0
    for i in range(len(classLabelAll)):
        if classLabelAll[i] == testLabel[i]:
            count = count + 1
        accuracy = count/len(testLabel)
    return accuracy

def printTree(tree, base_text):
    global cnt
    for key,value in tree.items():
        #print(key)
        if isinstance(tree[key],dict):
            base_text2 = ' Attrib : #%d:'%labels.index(key) if key in labels else ' %s;'%key
            printTree(value, base_text+base_text2)
        else:
            temp = '%s; %s'%(key, label_map[value])
            print('Branch[%d]:'%cnt + base_text +' '+ temp)
            cnt = cnt+1

if __name__=='__main__':
    dataSize, data_Increase,algo_type = None, None, None
    
    
    while dataSize not in [250,500,750,1000]:
        dataSize = int(input("Please enter training set size (must be a multiple of 250! and also <= 1000:)"))
        if dataSize < 0:
            print('Input Can not be negative')
        elif dataSize not in [250,500,750,1000]:
            print('Your input cannot be processed. Please enter a correct number')
        
    while data_Increase not in [10,25,50]:
        data_Increase = int(input("Please enter training set increment, must be either 10, 25, 50:"))
        if data_Increase < 0:
            print('Input Can not be negative')
        elif data_Increase not in [10,25,50]:
            print('Your input cannot be processed. Please enter a correct number')
    
    while algo_type not in ['I','C','c','i']:
        algo_type = str(input("Please enter a heuristic to use (either [C]ounting-based or [I]nformation theoretic): "))
        if algo_type not in ['I','C','c','i']:
            print('Your input cannot be processed. Please enter a correct selection. ["I","C"]')
    
    algo_m = {'C': 'Counting-based','I': 'Information theoretic'}
    print('\n\nPlease enter training set size (must be a multiple of 250! and also <= 1000:) : %d'%int(dataSize))
    print('Please enter training set increment, must be either 10, 25, 50: %d'%int(data_Increase))
    print('Please enter a heuristic to use (either [C]ounting-based or [I]nformation theoretic): ', algo_m[algo_type])
    
    X = 'data/input_files/mushroom_data.txt'
    trainingData, testData, labels = createDataSet1(X, int(dataSize), int(data_Increase))
    print('Train/Test Sizes : ', len(trainingData), len(testData))
    accuracies = []
    for i in list(range(0,int(dataSize)+1,int(data_Increase)))[1:]:
        trainingData2 = trainingData[:i]
        print('\nRunning With %d Examples for Training Set'%len(trainingData2))
        trainingCharacter, trainingLabel = splitData(trainingData2)
        testCharacter, testLabel = splitData(testData)  #
        featLabels = []
        labels2 = copy.deepcopy(labels)
        myTree = createTree(trainingData2, labels2, featLabels)
        classLabelAll = classifyAll(myTree, labels, testCharacter.tolist())
        accuracy = (np.array(testLabel) == np.array(classLabelAll)).mean()*100
        print('Given current tree, there are %d correct classifications out of %d possible (a success rate of %.4f percent).'
        %((np.array(testLabel) == np.array(classLabelAll)).sum(),len(testData), accuracy))
        accuracies.append(accuracy)
    # Compare Algorithms
    
    print()
    print()
    print('-------------STATS----------------\n')
    for i,j in enumerate(list(range(0,int(dataSize)+1,int(data_Increase)))[1:]):
        print('Training Set Size : %d, Success Rate : %.4f'%(j, accuracies[i]))
        
    print('\n\n-----------FINAL TREE-------------\n')
    print('Original Tree : ')
    print(json.dumps(myTree, indent=4))
    print()
    printTree(myTree,'')
    #print(myTree)
    fig = plt.figure(figsize=(8,6))
    fig.suptitle('Accuracy Comparison')
    plt.scatter(list(range(0,int(dataSize)+1,int(data_Increase)))[1:], accuracies, c='tab:red')
    plt.plot(list(range(0,int(dataSize)+1,int(data_Increase)))[1:], accuracies,c='tab:blue')
    plt.xlabel('Size of Training Set')
    plt.ylabel('% Correct')
    plt.savefig('out.png')
