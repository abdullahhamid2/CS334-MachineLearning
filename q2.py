#/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. Abdullah Hamid */
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()
dFrame = pd.DataFrame(iris.data, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
dFrame['species'] = iris.target
dFrame['species'] = dFrame['species'].map({0 : 'setosa' , 1 : 'versicolor' , 2 : 'virginica'})

def q2b():
    #boxplot for sepal length
    boxplot = sns.boxplot(x='species', y='sepal_length', data=dFrame)
    boxplot.set_xlabel("Species Type", fontsize=10)
    boxplot.set_ylabel("sepal length (cm)", fontsize=10)
    boxplot.get_figure().savefig('q2b - sepal length.png')
    #plt.clf()

    #boxplot for sepal width
    boxplot = sns.boxplot(x='species', y='sepal_width', data=dFrame)
    boxplot.set_xlabel("Species Type", fontsize=10)
    boxplot.set_ylabel("sepal width (cm)", fontsize=10)
    boxplot.get_figure().savefig('q2b - sepal width.png')
    plt.clf()

    #boxplot for petal length
    boxplot = sns.boxplot(x='species', y='petal_length', data=dFrame)
    boxplot.set_xlabel("Species Type", fontsize=10)
    boxplot.set_ylabel("petal length (cm)", fontsize=10)
    boxplot.get_figure().savefig('q2b - petal length.png')
    plt.clf()

    #boxplot for petal width
    boxplot = sns.boxplot(x='species', y='petal_width', data=dFrame)
    boxplot.set_xlabel("Species Type", fontsize=10)
    boxplot.set_ylabel("petal width (cm)", fontsize=10)
    boxplot.get_figure().savefig('q2b - petal width.png')
    plt.clf()

    #boxplot using pandas
    boxplot = dFrame.boxplot(column = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], by = 'species')

def q2c():
    #scatter plot for sepal
    sns.lmplot(x = 'sepal_length', y = 'sepal_width', data = dFrame, hue = 'species', fit_reg = False, legend = False)
    plt.legend()
    plt.savefig('q2c - sepal scatter.png')

    #scatter plot for petal
    sns.lmplot(x = 'petal_length', y = 'petal_width', data = dFrame, hue = 'species', fit_reg = False, legend = False)
    plt.legend()
    plt.savefig('q2c - petal scatter.png')

# q2d Based on Part B, Setosa's ca be identified by their small petal length and width. They also have a smaller sepal
# length paired with a longer sepal width based on Part C. Verisicolor and Virginica are harder to separate. Using the
# figure from Part B, Virginica's tend to have longer sepal length, petal length, and petal width. They also have the
# longest petal length-petal width pair (Figure from 2C). The rest are likely to be Verisicolor and have the middle
# values of petal length and width (Figure from 2B, 2C)
q2b()
q2c()
plt.show()