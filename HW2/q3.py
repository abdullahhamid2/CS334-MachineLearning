import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as skms


def hyper_parameter_find(classifier, xTrain, yTrain):
    averages = []
    for i in range(1, 48):
        if(classifier == "KNN"):
            dt = KNeighborsClassifier(n_neighbors=i)
        else:
            dt = DecisionTreeClassifier(max_depth=i, min_samples_leaf=10)
        scores = cross_val_score(dt, xTrain, yTrain.values.ravel(), cv=5, scoring='accuracy')
        averages.append(scores.mean())
    return averages.index(max(averages)) + 1

def param_to_dt(model, parameters, xTrain, yTrain, xTest, yTest):
    gs = skms.GridSearchCV(model, parameters, cv=5, scoring='roc_auc')
    result = {}
    print(gs.fit(xTrain, yTrain).best_params_)
    for index in np.arange(float("{:.2f}".format(0.05)), float("{:.2f}".format(0.25)), float("{:.2f}".format(0.05))):
        new_index = np.random.choice(len(xTrain), int(len(xTrain) * (1 - index)), replace=False)
        x_train_new = xTrain.iloc[new_index, :]
        y_train_new = yTrain[new_index]
        final = train_test(gs.fit(xTrain, yTrain).best_estimator_, x_train_new, y_train_new, xTest, yTest)
        result[float("{:.2f}".format(index))] = final
    return pd.DataFrame(result)

def train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn train/test calculate the auc
    """
    model.fit(xTrain, yTrain)
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    fpr, tpr, thresholds = metrics.roc_curve(yTrain, yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(yTest, yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    yHatClassTrain = model.predict(xTrain)
    yHatClassTest = model.predict(xTest)
    return {"trainAuc": trainAuc, "testAuc": testAuc, "trainAcc": metrics.accuracy_score(yTrain, yHatClassTrain),
            "testAcc": metrics.accuracy_score(yTest, yHatClassTest)}


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    classifier = "KNN"
    print("Optimal hyperparameter for knn algorithm is: %s" % hyper_parameter_find(classifier,xTrain, yTrain))
    classifier = "DT"
    print("Optimal hyperparameter for Decision Tree algorithm is: %s" % hyper_parameter_find(classifier, xTrain, yTrain))
    yTrain = yTrain.to_numpy().flatten()
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    yTest = yTest.to_numpy().flatten()

    KNN_PARAMETERS = {'n_neighbors': range(1, 48, 2)}
    DT_PARAMETERS = {'min_samples_leaf': range(5, 20, 5),
                     'max_depth': range(3, 10, 2),
                     'criterion': ['gini', 'entropy']}
    print("KNN Results------------------------------------------------------------------")
    print(param_to_dt(KNeighborsClassifier(), KNN_PARAMETERS, xTrain, yTrain, xTest, yTest).to_markdown())
    print("DT Results--------------------------------------------------------------------")
    print(param_to_dt(DecisionTreeClassifier(), DT_PARAMETERS, xTrain, yTrain, xTest, yTest).to_markdown())

if __name__ == "__main__":
    main()
