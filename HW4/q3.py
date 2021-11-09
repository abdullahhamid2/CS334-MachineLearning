import argparse
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from perceptron import file_to_numpy, calc_mistakes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def bayes (xFeat, y, xTest, yTest):
	print(type(y))
	print(y)
	#getting rid of the indices
	if (y.ndim > 1):
		y = np.delete(y, 0, 1)
	if (yTest.ndim > 1):
		yTest = np.delete(yTest, 0, 1)
	#xFeat = np.delete(xFeat, 0, 1)


	#print(len(xFeat))
	gnb = GaussianNB()
	print(xFeat)
	gnb.fit(xFeat, y.ravel())
	y_pred = gnb.predict(xTest)
	#print(y_pred)
	#print(yTest)
	#print(accuracy_score(y_pred, yTest))
	return y_pred


def linreg (xFeat, y, xTest, yTest):
	#getting ird of the indices
	if (y.ndim > 1):
		y = np.delete(y, 0, 1)
	if (yTest.ndim > 1):
		yTest = np.delete(yTest,0,1)
	#xFeat = np.delete(xFeat, 0, 1)


	lg = LogisticRegression()
	lg.fit(xFeat, y.ravel())
	y_pred = lg.predict(xTest)
	#print(y_pred)
	#print(yTest)
	#print(accuracy_score(y_pred, yTest))
	return y_pred



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("xTrain",
                        default="binary.csv",
                        help="filename for features of the training data")
	parser.add_argument("yTrain",
                        default = "y.csv",
                        help="filename for labels associated with training data")
	parser.add_argument("xTest",
                        default = "binary.csv",
                        help="filename for features of the test data")
	parser.add_argument("yTest",
                        default = "y.csv",
                        help="filename for labels associated with the test data")
	parser.add_argument("--seed", default=334, type=int, help="default seed number")
	args = parser.parse_args()
	# load the train and test data assumes you'll use numpy
	xTrain = file_to_numpy(args.xTrain)
	xTest = file_to_numpy(args.xTest)
	yTrain = file_to_numpy(args.yTrain)
	yTest = file_to_numpy(args.yTest)
	bay_pred = bayes(xTrain, yTrain, xTest, yTest)
	lin_pred = linreg (xTrain, yTrain, xTest, yTest)
	yTest = np.delete(yTest,0,1)
	print ("This Bayesian prediction is: " + str(bay_pred))
	print ("This linear regression prediction is: " + str(lin_pred))
	print(accuracy_score(bay_pred, yTest))
	print(accuracy_score(lin_pred, yTest))


if __name__ == "__main__":
	main()