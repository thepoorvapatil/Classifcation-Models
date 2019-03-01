#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:01:13 2018

@author: poorvapatil
"""
import csv
import random
import math

#get csv file
def loadCSV(filename):
    
    x=csv.reader(open(filename,"r"))
    dataset=list(x)
    dataset.pop(0) # remove the attributes heading from the dataset
    #remove UID and change M/F to 2/4 in train and test
    for x in dataset:
        x.pop(0)
        if x[0]=='Male':
            x[0]='2'
        else:
            x[0]='4'
    return dataset

#split into training and testing values
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    print(trainSize)
    for x in range (0,trainSize):
        trainSet.append(copy.pop(x)) #append first 70% of the data
    
    '''
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy)) #pick randomly
        trainSet.append(copy.pop(index)) #append random data in train
    '''
    
    return [trainSet, copy]  #trainset has train values, copy has test vals


#separate by 0s and 1s from data
def separateByClass(dataset):
	separated = {} #make a dictionary to separate 0s and 1s
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

#get mean of values
def mean(numbers):
    numbs=list(numbers)
    for i in range(len(numbs)):
        numbs[i]=int(numbs[i])
    return sum(numbs)/float(len(numbers))

#get standard dev 
def stdev(numbers):
    avg = mean(numbers)
    numbs=list(numbers)
    for i in range(len(numbs)):
        numbs[i]=int(numbs[i])
    variance = sum([pow((x-avg),2) for x in numbs])/float(len(numbs)-1)
    return math.sqrt(variance)

#get the mean and stddev for each attribute in a new list called summaries
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

#for 0/1, separate the mean and stddev which calls summarize() to find for each attribute
def summarizeByClass(dataset):
    separated = separateByClass(dataset) #returns a dict
    summaries = {} 
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

#use gaussian NB formula to get P(x|y)
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))) 
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x=inputVector[i]
            probabilities[classValue] *= calculateProbability(float(x), float(mean), float(stdev))
    return probabilities


def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions


def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



splitRatio=0.7
filename="/Users/poorvapatil/Downloads/Employee_data.csv"
dataset = loadCSV(filename) #does not contain headings or UIDs, with M/F changed to 2/4
train, test = splitDataset(dataset, splitRatio)
'''
print(len(dataset))
print(len(train))
print(len(test))
print()
'''
print('DATASET=', len(dataset))
print('SplitRatio=',splitRatio*100,'%')
print('Training set=',len(train))
print('Testing set=',len(test))
summaries = summarizeByClass(train)
predictions = getPredictions(summaries, test)
accuracy = getAccuracy(test, predictions)
print('Accuracy:',accuracy,'%')

