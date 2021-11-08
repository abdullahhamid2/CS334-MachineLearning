#/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. ABDULLAH HAMID */
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # TODO do more than this
    day_refactor = pd.to_datetime(df['date']).dt.dayofyear
    date_refactor = pd.to_datetime(df['date'])
    time_df = (date_refactor.dt.hour * 60) + date_refactor.dt.minute
    df.insert(loc=0, column='time', value=time_df)
    df.insert(loc=0, column='day_of_year', value=day_refactor)
    df = df.drop(columns=['date'])
    return df

def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # TODO
    df = df[['time', 'lights', 'T2', 'RH_out']]
    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    scaledtrain = MinMaxScaler().fit_transform((trainDF))
    scaledtest = MinMaxScaler().fit_transform((testDF))
    trainDF = pd.DataFrame(scaledtrain, index=trainDF.index, columns=trainDF.columns)
    testDF = pd.DataFrame(scaledtest, index=testDF.index, columns=testDF.columns)

    return trainDF, testDF

def plot(xTrain, yTrain):
    merger = pd.concat([xTrain, yTrain], axis=1)
    corr = merger.corr(method='pearson')
    plotter = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="RdBu")
    plt.show()

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)

    xTrainPlot = pd.read_csv('new_xTrain.csv')
    yTrainPlot = pd.read_csv('eng_yTrain.csv')

    plot(xTrainPlot, yTrainPlot)

if __name__ == "__main__":
    main()
