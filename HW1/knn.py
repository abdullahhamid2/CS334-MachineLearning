import argparse
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
#/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. Abdullah Hamid */


class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k):
        """
        Knn constructor
        Parameters
        ----------
        k : int
            Number of neighbors to use.
        """
        self.k = k
        self.features = []
        self.labels = []
    def train(self, xFeat, y):
        """
        Train the k-nn model.
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
        # TODO do whatever you need
        if type(xFeat) != np.ndarray:  # if the data isn't a numpy array, eg dataframe, convert to numpy
            self.features = xFeat.to_numpy()
        else:
            self.features = xFeat
        self.labels = y
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
        # TODO
        if type(xFeat) != np.ndarray:           #convert to numpy
            xFeat = xFeat.to_numpy()
        x = 0
        for testfeatures in xFeat:                      # for each row of test data
            distances = list()
            i = 0
            for row in self.features:             # iterate through each row of training data
                sums = 0
                for index in range(len(testfeatures)):  # calculate euclidean distance
                    #sums += math.pow(abs(testfeatures[index] - row[index]), len(testfeatures))
                    #sums = math.pow(sums,0.5)
                    sums += euclidean_distance(testfeatures[index], row[index], 2)
                distances.append((self.labels[i], sums))
                i += 1
            distances.sort(key=lambda tup: tup[1])
            output = distances
            majority_vote = 0
            for index in range(self.k):
                if output[index][0] == 1.0:   #look into the outputs and count the 1's and 0's and find the majority winner
                    majority_vote += 1
                else:
                    majority_vote -= 1
            if majority_vote >= 0:
                yHat.append([1])
            else:
                yHat.append([0])
            x += 1
        return yHat

def euclidean_distance(n1,n2,x):   #calculating euclidean distance. 
    dist = 0
    dist += math.pow(abs((n1 - n2)), x)
    #dist = math.pow(dist, 0.5)
    return dist

def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction
    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples
    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0
    length = len(yHat)
    for index in range(length):
        if yHat[index] == yTrue[index]:
            acc += 1
    acc = (acc / len(yTrue))
    return acc           # return (num correct / total test samples)

def plot_data(k, x_features, y_labels, x_testing, y_testing):
    plotter = np.empty([k, 2])
    for i in range(1, k):             # for i=1 to k
        knn = Knn(i)
        knn.train(x_features, y_labels['label'])
        label_trainer = knn.predict(x_features)
        trainer_accuracy = accuracy(label_trainer, y_labels['label'])
        y_hat_test = knn.predict(x_testing)
        test_acc = accuracy(y_hat_test, y_testing['label'])
        plotter[i - 1][0] = trainer_accuracy
        plotter[i - 1][1] = test_acc
    # setting up plot
    plt.title("Training and Testing Accuracy(KNN Algorithm)")
    plt.ylabel("Accuracy")
    plt.xlabel("K(input)")
    plt.plot([i for i in range(1, k+1)], plotter)       # start plot from 1, not 0 (default)
    plt.legend(("Training Accuracy", "Testing Accuracy"))
    plt.show()
def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    plot_data(args.k, xTrain, yTrain, xTest, yTest)
    #plot_data(knn, trainAcc, testAcc)
if __name__ == "__main__":
    main()