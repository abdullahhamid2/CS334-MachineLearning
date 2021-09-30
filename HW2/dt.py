#/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. Abdullah Hamid */

import argparse
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def split(xFeat, y, split_feat, split_value):
    set = pd.concat([xFeat, y], axis=1)
    lX = set[set[split_feat] <= split_value]
    lY = lX.iloc[:, -1]
    lX = lX.drop(labels='label', axis=1)
    rX = set[set[split_feat] > split_value]
    rY = rX.iloc[:, -1]
    rX = rX.drop(labels='label', axis=1)
    return lX, lY, rX, rY

def leaf_test(tree, sample):
    if tree.right is None and tree.left is None:
        return tree.classification
    elif tree.right is not None and sample[tree.featName] > tree.featVal:
        return leaf_test(tree.right, sample)
    elif tree.left is not None and sample[tree.featName] <= tree.featVal:
        return leaf_test(tree.left, sample)
    return tree.classification

def get_gini(y):  # calculates the gini index
    gini = 0
    for i in range(len(y.value_counts().values)):
        gini += pow((y.value_counts().values[i] / len(y)) , 2)
    gini = 1 - gini
    return gini

def get_entropy(y):  # calculates entropy
    entropy = 0
    for i in range(len(y.value_counts().values)):
        val = math.log2((y.value_counts().values[i] / len(y)))  # do log2 of the fraction
        entropy = entropy + val
    entropy = -1 * entropy
    return entropy

class Node:
    left = None
    right = None
    features = None
    labels = None
    classification = None
    featName = None
    featVal = None

    def __init__(self, features, labels, left, right, classification, featName, featVal):
        self.left = left
        self.right = right
        self.features = features
        self.labels = labels
        self.classification = classification
        self.featName = featName
        self.featVal = featVal

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    tree = None
    depth = 0
    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor
        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int
            Maximum depth of the decision tree
        minLeafSample : int
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.tree = None

    def grow_tree(self, xFeat, y, depth):
        length = y.value_counts()
        classification = length.idxmax()
        minLeaves = self.minLeafSample
        maxDepth = self.maxDepth
        featLength = len(xFeat)
        finalValue, finalGini, finalEntropy, finalFeature = 0, 1000, 1000, None  # Integer_MAX_VALUE?
        if (len(length) == 1 or depth >= maxDepth or featLength<minLeaves):
            return Node(xFeat, y, None, None, classification, None, None)
        else:
            for featName, feature in xFeat.iteritems():
                for value in xFeat[featName].unique():
                    leftX, leftY, rightX, rightY = split(xFeat, y, featName, value)
                    if self.criterion == 'gini':
                        leftGini, rightGini = get_gini(leftY), get_gini(rightY)
                        averageG = (leftGini * len(leftY)
                                    + rightGini * len(rightY)) / len(y)
                        if(averageG < finalGini and averageG<1):
                            finalGini, finalValue, finalFeature = averageG, value, featName
                    elif self.criterion == 'entropy':
                        entropy = (len(leftY)/len(y) * get_entropy(leftY))\
                                + (len(rightY)/len(y) * get_entropy(rightY))
                        if(entropy < finalEntropy and entropy>0):
                            finalValue, finalFeature, finalEntropy = value, featName, entropy
            tree = Node(xFeat, y, None, None, y.value_counts().idxmax(), finalFeature, finalValue)
            leftX, leftY, rightX, rightY = split(xFeat, y, finalFeature, finalValue)
            tree.left = self.grow_tree(leftX, leftY, depth + 1)
            tree.right = self.grow_tree(rightX, rightY, depth + 1)
        return tree

    def train(self, xFeat, y):
        """
        Train the decision tree model.
        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of labels associated with training data.
        Returns
        -------
        self : object
        """
        self.tree = self.grow_tree(xFeat, y, 0)
        return self

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict
        what class the values will have.
        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.
        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """

        yHat = [] # variable to store the estimated class label
        for index, sample in xFeat.iterrows():
            yHat.append(leaf_test(self.tree, sample))             # predict the value
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.
    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.
    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])

    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc
def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
