#/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. ABDULLAH HAMID */
import argparse
import numpy as np
import pandas as pd
import time
from standardLR import StandardLR
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from lr import LinearRegression, file_to_numpy

class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    # def new_batch(self, x, y):
    #     for i in np.arange(0, x.shape[0], self.bs):
    #         yield x[i:i + self.bs], y[i:i + self.bs]

    def train_predict(self, xTrain, yTrain, xTest, yTest, split_factor = 1.0):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD
        length0 = len(xTrain[0])
        lengthTrain = len(xTrain)
        lengthTest = len(xTest)
        self.beta = np.ones(length0 + 1)
        row = np.ones(lengthTrain)
        rowtransposed = np.ones(lengthTest)
        X = np.concatenate((row[:, np.newaxis], xTrain), axis=1)
        Xtransposed = np.concatenate((rowtransposed[:, np.newaxis], xTest), axis=1)

        if split_factor != 1.0:
            criteriaX = int(len(X) * split_factor)
            criteriaY = int(len(yTrain) * split_factor)
            X, removedX = np.split(X, criteriaX)
            yTrain, removedyTrain = np.split(yTrain, criteriaY)

        for epoch in range(self.mEpoch):
            start = time.time()
            merged_set = np.concatenate((X, yTrain), axis=1)
            merged_set = shuffle(merged_set)
            new_x_train = merged_set[:, :-1]
            new_y_train = merged_set[:, -1]
            lenX = len(X) / self.bs
            lenY = len(yTrain) / self.bs
            x_batches = np.array_split(new_x_train, lenX)
            y_batches = np.array_split(new_y_train, lenY)
            for x_batch, y_batch in zip(x_batches, y_batches):
                gradient = np.zeros(5)
                for sample_x, sample_y in zip(x_batch, y_batch):
                    sampler  = sample_y - np.matmul(sample_x, self.beta)
                    new_grad = np.multiply(np.transpose(sample_x), sampler)
                    gradient = np.add(new_grad, gradient)
                avg_gradient = np.divide(gradient, self.bs)
                self.beta = self.beta + (self.lr * avg_gradient)
            end = time.time()
            trainStats[len(x_batches) * epoch] = {'time': end - start, 'train-mse': self.mse(X, yTrain), 'test-mse': self.mse(Xtransposed, yTest)}
        return trainStats

def plot3B(xTrain, yTrain, xTest, yTest,epoch):
    for lr in [0.1, 0.01, 0.001, 0.0001, .00001]:
        model = SgdLR(lr, 1, epoch)
        trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
        results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
        plt.plot(results, label=lr)

    plt.title("Training and Test MSE(LR = 0.001, BS = 1, #EPOCHS = 100)")
    plt.ylim(top=1.25, bottom=.3)
    plt.xlabel("#Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def plot3C(xTrain, yTrain, xTest, yTest,epoch):
    model = SgdLR(.001, 1, epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
    plt.plot(results, label='train-mse-lr=.001')
    results = [trainStats[i]['test-mse'] for i in trainStats.keys()]
    plt.plot(results, label='test-mse-lr=.001')
    plt.title("Training and Test MSE(LR = 0.001, BS = 1, #EPOCHS = 100)")
    plt.xlabel("#Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def plot4A(xTrain,yTrain,xTest,yTest,epoch):
    batch_list = [1,20,40,60,80,120,180,210,240,len(xTrain)]
    # for index in batch_list:
    #     plt.figure()
    #     plt.title("Training and Testing for different EPOCH values")
    #     plt.xlabel("#EPOCH")
    #     plt.ylabel("MSE")
    #     for lr in [0.1,0.01,0.001,0.0001]:
    #         model = SgdLR(lr, index, epoch)
    #         train_stats  = model.train_predict(xTrain,yTrain, xTest, yTest)
    #         results = [train_stats[i]['train-mse'] for i in train_stats.keys()]
    #         plt.ylim(top=2.0)
    #         plt.plot(results, label="bs=" + str(index) + " lr=" + str(lr))
    #     plt.legend()
    # plt.show()

    optimal_lr = [.001, .01, .01, .01, .1,.1, .1, .1, .1, .1]    # manually selected from above plots
    train_mse = []
    test_mse = []
    time_list = []
    for batch_size, lr in zip(batch_list, optimal_lr):                 # for each batch size and optimal lr
        start = time.time()                                                 # train a model
        model = SgdLR(lr, batch_size, epoch)
        train_stats = model.train_predict(xTrain, yTrain, xTest, yTest)
        end = time.time()
        train_mse.append(train_stats[list(train_stats.keys())[-1]]['train-mse'])  # save the ending train, test mse and
        test_mse.append(train_stats[list(train_stats.keys())[-1]]['test-mse'])    # time it took
        time_list.append(end-start)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)              # train a closed form model

    # print out two scatter plots, one for train mse, one for test-mse vs time taken.

    plt.figure()
    plt.title("Training MSE vs Total Time for 50 Epochs")
    plt.xlabel("Time (sec)")
    plt.ylabel("Training MSE")
    plt.scatter(time_list, train_mse, label='sgd-mse')                                          # SGD points
    plt.scatter(trainStats[0]['time'], trainStats[0]['train-mse'], label='closed-form-mse')     # closed form point
    plt.ylim(top=.5, bottom=.25)                                                # limit y axis to better see the data

    plt.figure()
    plt.title("Test MSE vs Total Time for 50 Epochs")
    plt.xlabel("Time (sec)")
    plt.ylabel("Test MSE")
    plt.scatter(time_list, test_mse, label='sgd-mse')                                           # SGD points
    plt.scatter(trainStats[0]['time'], trainStats[0]['test-mse'], label='closed-form-mse')      # closed form point
    plt.ylim(top=.5, bottom=.25)                                                # limit y axis to better see the data
    plt.legend()
    plt.show()


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)
    epoch = args.epoch
    #plot3B(xTrain,yTrain,xTest,yTest,epoch)
    #plot3C(xTrain,yTrain,xTest,yTest,epoch)
    plot4A(xTrain,yTrain,xTest,yTest,epoch)

if __name__ == "__main__":
    main()

