"""
main code section for the kNN classifier
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import random

class kNNClassifier:
    #class variable for stroing valid distance metrics
    distanceMetrics = {"Manhattan", "Euclidean"}

    #constructor takes k hyperparaneter and distance metric as a string
    def __init__(self, k, distanceMetric, classes):

        self._k = k
        self._distanceMetric = distanceMetric
        self.Data = pd.DataFrame()
        self._classes = classes

    def setK(self, newK):
        self._k = newK
    
    def LoadTraining(self, Path, Label):
        """
        Loads the CSV Data into the classifier from a path to a csv file
        """
        df = pd.read_csv(Path, names=["X", "Y"])
        x =  df.shape[0]
        LabelCol = [Label] * x
        df["Label"] = LabelCol
        if(not self.Data.empty):
            dataJoined = []
            for item in self.Data.values:
                dataJoined.append(item)
            for item in df.values:
                dataJoined.append(item)
            self.Data = pd.DataFrame(dataJoined)
        else:
            self.Data = df
        
    
    
    def classifyPoint(self, point):
        """
        classifies a point using the kNearest Neighbors
        this is done by naively calculating the distance between the given point and the rest of the dataset
        """
        nearestNeighbors = dict()
        for dataPoint in self.Data.values:
            if(self._distanceMetric == "Euclidian"):
                
                distance = self.distanceEuclidean(point, (dataPoint[0], dataPoint[1]))
            else:
                distance = self.distanceManhattan(point, (dataPoint[0], dataPoint[1]))
            #assuming that the point is different
            if(distance > 0):
                if len(nearestNeighbors) < self._k:
                    nearestNeighbors[distance] = dataPoint
                else:
                    if distance < max(nearestNeighbors.keys()):
                        nearestNeighborsKeySorted = sorted(nearestNeighbors.keys(), reverse=True)
                        for item in nearestNeighborsKeySorted:
                            if distance == item:
                                randval = random.randint(0,1)
                                if randval == 1:
                                    nearestNeighbors[distance] = dataPoint
                                    nearestNeighbors.pop(item)
                            elif distance < item:
                                nearestNeighbors[distance] = dataPoint
                                nearestNeighbors.pop(item)
                                break

        classcounts = dict()
        for c in self._classes:
            classcounts[c] = 0

        for item in nearestNeighbors.values():
            label = item[2]
            if(classcounts.__contains__(label)):
                classcounts[label] += 1
        result = max(classcounts, key=classcounts.get)
        return result

        
    

    def distanceManhattan(self,p1, p2):
        """
        given 2 points, compute the manhattan distance between them
        """
        distance = (abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]))
        return distance
    
    def distanceEuclidean(self, p1, p2):
        distance = pow(pow((p2[0] - p1[0]),2) + pow((p2[1] - p1[1],2)),0.5)
        return distance

    #display the data for the classifier, using a provided grid coords set for boundary drawing and an optional test set of points
    def displayData(self, gridCoords, Figure, testSet = None):
        gridDFClassified = self.classifyPointList(gridCoords)
        
        #creating the boundary drawing scatter plots
        #sNC bound region
        boundaryClass1 = [[],[]]
        #sDAT bound region
        boundaryClass2 = [[],[]]
        for row in range(gridDFClassified.shape[0]):
            if(gridDFClassified.values[row][2] == '0'):
                boundaryClass1[0].append(gridDFClassified.values[row][0])
                boundaryClass1[1].append(gridDFClassified.values[row][1])
            else:
                boundaryClass2[0].append(gridDFClassified.values[row][0])
                boundaryClass2[1].append(gridDFClassified.values[row][1])

        #configuring the training data into plottable formats
        #TrainingData sNC set
        tDataClass1 = [[],[]]
        #TrainingData sDAT set
        tDataClass2 = [[],[]]

        for row in range(self.Data.shape[0]):
            if(self.Data.values[row][2] == '0'):
                tDataClass1[0].append(self.Data.values[row][0])
                tDataClass1[1].append(self.Data.values[row][1])
            else:
                tDataClass2[0].append(self.Data.values[row][0])
                tDataClass2[1].append(self.Data.values[row][1])

        #if a test set is also provided, prepare it for plotting as well
        if testSet != None:
            #testr data snc set
            testDataClass1 = [[],[]]
            #test data sdat set
            testDataClass2 = [[],[]]
            for row in range(testSet.shape[0]):
            if(testSet.values[row][2] == '0'):
                testDataClass1[0].append(testSet.values[row][0])
                testDataClass1[1].append(testSet.values[row][1])
            else:
                testDataClass2[0].append(testSet.values[row][0])
                testDataClass2[1].append(testSet.values[row][1])

        plt.figure()
        #plotting the sNC data region
        plt.scatter(boundaryClass1[0],boundaryClass1[1], 1.0, 'green')
        #plotting the sDat data region
        plt.scatter(boundaryClass2[0],boundaryClass2[1], 1.0, 'blue')
        #plotting the sNC training points
        plt.scatter(tDataClass1[0], tDataClass1[1], 3, 'green',marker = "D")
        #plotting the sDat training points
        plt.scatter(tDataClass2[0], tDataClass2[1], 3, 'blue', marker = "D")
        if(testSet != None):
             plt.scatter(testDataClass1[0], testDataClass1[1], 3, 'green',marker = "X")
        #plotting the sDat training points
             plt.scatter(testDataClass2[0], testDataClass2[1], 3, 'blue', marker = "X")
        
        plt.show()

    def classifyPointList(self, pointList, evaluationLabels=None):
        
        """
        Loads a list of points from a csv path into a pd datframe, and labels them using the classify point function
        if a set of evaluation labels is provided, the system will also evaluate and return accuracy
        Returns a pandas dataframe containing the points and their labels
        """
        classifications = []
        df = pd.read_csv(pointList, names=["X", "Y"])
        for point in df.values:
            classifications.append(self.classifyPoint((point[0], point[1])))
        df["Label"] = classifications
        if(evaluationLabels != None):
            correctLabelCount = 0
            for row in range(df.shape[0]):
                trainLabel = df.values[row][2]
                testLabel = evaluationLabels[row]
                if(trainLabel == testLabel):
                    correctLabelCount += 1
            accuracy = correctLabelCount / df.shape[0]
            return df, accuracy

        else:
            return df

    
        

if __name__ == "__main__":
    print("Debug Section for Classifier Class")

    #create a test class of the knn classifier
    testClass = kNNClassifier(10, "Euclidean",["0", "1"])

    #load some sample training data
    testClass.LoadTraining("Assignment 1/train.sDAT.csv", '1')
    testClass.LoadTraining('Assignment 1/train.sNC.csv', '0')
    
    

    kTestVals = [1,3,5,10,20,30,50,100,150,200]

    testErrVals = []
    trainErrVals = []

    ConcatDatas = []

    #runs a loop through the assignment required k vals and tests them all 
    for k in kTestVals:
        testClass.setK(k)
        #run tests on the snc and sdat csv files and record accuracy
        trainDATLabeled, trainacc1 = testClass.classifyPointList("Assignment 1/train.sDAT.csv", (['1'] * 237))
        trainNCLabeled, trainacc2 = testClass.classifyPointList("Assignment 1/train.sNC.csv", (['0'] * 237))

        testDATLabeled, testacc1 = testClass.classifyPointList("Assignment 1/test.sDAT.csv", (['1'] * 100))
        testNCLabeled, testacc2 = testClass.classifyPointList("Assignment 1/test.sNC.csv", (['0'] * 100))
        testaccTotal = (testacc1 + testacc2) / 2
        trainaccTotal = (trainacc1 + trainacc2) / 2

        testErrVals.append(1 - testaccTotal)
        trainErrVals.append(1 - trainaccTotal)
        print("K = ", k, "train error = ", trainaccTotal, "|test error = ", testaccTotal)

        testDataListConcat = []
        for row in range(testDATLabeled.shape[0]):
            testDataListConcat.append(testDATLabeled.values[row])
        for row in range(testNCLabeled.shape[0]):
            testDataListConcat.append(testNCLabeled.values[row])
        testConcatDF = pd.DataFrame(testDataListConcat)

        ConcatDatas.append(testConcatDF)

    
    fig = plt.figure()
    for graphData in ConcatDatas:

        testClass.displayData('Assignment 1/2D_grid_points.csv', fig, graphData)

    plt.subplot(627, kTestVals, trainErrVals)
    plt.subplot(627, kTestVals, testErrVals)

    
    done = input("press key when finished")
