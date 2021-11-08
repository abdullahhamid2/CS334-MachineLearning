#/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. ABDULLAH HAMID */
import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SOMETHING
        start = time.time()

        column0 = np.ones(len(xTrain))
        column1 = np.ones(len(xTest))

        list0 = np.concatenate((column0[:, np.newaxis], xTrain), axis=1)
        list1 = np.concatenate((column1[:, np.newaxis], xTest), axis=1)

        matmul = np.matmul(np.linalg.inv(np.matmul(np.transpose(list0), list0)), np.transpose(list0))
        self.beta = np.matmul(matmul, yTrain)
        end = time.time()

        trainStats[0] = {'time': end-start, 'train-mse': self.mse(list0, yTrain), 'test-mse': self.mse(list1, yTest)}
        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
