import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from load_data import *

# HELPER FUNCTIONS
def convertToClass(Y, k):
    return np.apply_along_axis(lambda elem : np.round(elem * k / 5), 0, Y)

def evaluateRegressionModel(model, x, y):
    y_pred = np.clip(model.predict(x), 0, 5).flatten()
    
    # MSE
    # print("Mean Square Error:")
    mse = mean_squared_error(y, y_pred)
    # print(mse)
    
    # Accuracy
    # print("Accuracy:")
    accuracy = sum(np.abs(y_pred - y.values.flatten()) <  0.25)/ y.shape[0]
    # print(accuracy)
    
    # Confusion matrix
    y_pred = convertToClass(y_pred, 10)
    y_actual = convertToClass(y, 10)
    confusion_m = confusion_matrix(y_actual, y_pred)
    # print("Confusion Matrix:")
    # print(confusion_m)
    # print("")
    return (mse, accuracy, confusion_m)

# MODELS
def evaluateClassificationModel(model, x, y):
    y_pred = model.predict(x)
    confusion_m = confusion_matrix(y, y_pred)
    accuracy = model.score(x, y)
    # print("Accuracy:")
    # print(accuracy)
    # print("Confusion Matrix:")
    # print(confusion_m)
    # print("")
    return (accuracy, confusion_m)

def linearRegression_L1(x_train, y_train, x_dev, y_dev, x_test, y_test, alpha=0.1):
	reg = linear_model.Lasso(alpha=alpha)
	reg.fit(x_train, y_train)
	print("LINEAR REGRESSION L1")
	mse_train, accuracy_train, confusion_m_train = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	print("training MSE: " + str(mse_train))

	mse_dev, accuracy_dev, confusion_m_dev = evaluateRegressionModel(reg, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	print("dev MSE: " + str(mse_dev))

	mse_test, accuracy_test, confusion_m_test = evaluateRegressionModel(reg, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))
	print("test MSE: " + str(mse_test))

	return reg, mse_train, accuracy_train, confusion_m_train, mse_dev, accuracy_dev, confusion_m_dev, mse_test, accuracy_test, confusion_m_test

def linearRegression_L2(x_train, y_train, x_dev, y_dev, x_test, y_test, alpha=1.0):
	reg = linear_model.Ridge(alpha=alpha)
	reg.fit(x_train, y_train)
	print("LINEAR REGRESSION L2")
	mse_train, accuracy_train, confusion_m_train = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	print("training MSE: " + str(mse_train))

	mse_dev, accuracy_dev, confusion_m_dev = evaluateRegressionModel(reg, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	print("dev MSE: " + str(mse_dev))

	mse_test, accuracy_test, confusion_m_test = evaluateRegressionModel(reg, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))
	print("test MSE: " + str(mse_test))
	return reg, mse_train, accuracy_train, confusion_m_train, mse_dev, accuracy_dev, confusion_m_dev, mse_test, accuracy_test, confusion_m_test

def linearRegression(x_train, y_train, x_dev, y_dev, x_test, y_test):
	reg = LinearRegression()
	reg.fit(x_train, y_train)
	print("LINEAR REGRESSION")
	mse_train, accuracy_train, confusion_m_train = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	print("training MSE: " + str(mse_train))

	mse_dev, accuracy_dev, confusion_m_dev = evaluateRegressionModel(reg, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	print("dev MSE: " + str(mse_dev))

	mse_test, accuracy_test, confusion_m_test = evaluateRegressionModel(reg, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))
	print("test MSE: " + str(mse_test))
	return reg, mse_train, accuracy_train, confusion_m_train, mse_dev, accuracy_dev, confusion_m_dev, mse_test, accuracy_test, confusion_m_test

def supportVectorRegression(x_train, y_train, x_dev, y_dev, x_test, y_test, C=0.1, kernel='rbf', epsilon=0.1, shrinking=True):
	reg = SVR(C=C, kernel=kernel, epsilon=epsilon, shrinking=shrinking)
	reg.fit(x_train, y_train)
	print("SUPPORT VECTOR REGRESSION")
	mse_train, accuracy_train, confusion_m_train = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	print("training MSE: " + str(mse_train))

	mse_dev, accuracy_dev, confusion_m_dev = evaluateRegressionModel(reg, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	print("dev MSE: " + str(mse_dev))

	mse_test, accuracy_test, confusion_m_test = evaluateRegressionModel(reg, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))
	print("test MSE: " + str(mse_test))
	return reg, mse_train, accuracy_train, confusion_m_train, mse_dev, accuracy_dev, confusion_m_dev, mse_test, accuracy_test, confusion_m_test

def supportVectorRegressionPCA(x_train, y_train, x_dev, y_dev, x_test, y_test, C=0.1, kernel='rbf', epsilon=0.1, shrinking=True, n_components=5):
	pca = PCA(n_components=n_components)
	x_train = pca.fit_transform(x_train)
	x_test = pca.transform(x_test)
	
	reg = SVR(C=C, kernel=kernel, epsilon=epsilon, shrinking=shrinking)
	reg.fit(x_train, y_train)
	print("SUPPORT VECTOR REGRESSION WITH PCA")
	mse_train, accuracy_train, confusion_m_train = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	print("training MSE: " + str(mse_train))

	mse_dev, accuracy_dev, confusion_m_dev = evaluateRegressionModel(reg, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	print("dev MSE: " + str(mse_dev))

	mse_test, accuracy_test, confusion_m_test = evaluateRegressionModel(reg, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))
	print("test MSE: " + str(mse_test))
	return reg, mse_train, accuracy_train, confusion_m_train, mse_dev, accuracy_dev, confusion_m_dev, mse_test, accuracy_test, confusion_m_test

def softmax(x_train, y_train, x_dev, y_dev, x_test, y_test, penalty='l2', C=1.0):
	clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=C)
	clf.fit(x_train, y_train)
	print("SOFTMAX")
	accuracy_train, confusion_m_train = evaluateClassificationModel(clf, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	accuracy_dev, confusion_m_dev = evaluateClassificationModel(clf, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	accuracy_test, confusion_m_test = evaluateClassificationModel(clf, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))

	return clf, accuracy_train, confusion_m_train, accuracy_dev, confusion_m_dev, accuracy_test, confusion_m_test

def svm(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'auto', kernel='rbf', C=.8):
	clf = SVC(gamma = gamma, kernel=kernel, C=C)
	clf.fit(x_train, y_train)
	print("SUPPORT VECTOR MACHINE")
	accuracy_train, confusion_m_train = evaluateClassificationModel(clf, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	accuracy_dev, confusion_m_dev = evaluateClassificationModel(clf, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	accuracy_test, confusion_m_test = evaluateClassificationModel(clf, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))

	return clf, accuracy_train, confusion_m_train, accuracy_dev, confusion_m_dev, accuracy_test, confusion_m_test


def decisionTree(x_train, y_train, x_dev, y_dev, x_test, y_test, max_depth = 15):
	clf = tree.DecisionTreeClassifier(max_depth = max_depth)
	clf.fit(x_train, y_train)
	print("DECISION TREE")
	accuracy_train, confusion_m_train = evaluateClassificationModel(clf, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	accuracy_dev, confusion_m_dev = evaluateClassificationModel(clf, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	accuracy_test, confusion_m_test = evaluateClassificationModel(clf, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))

	return clf, accuracy_train, confusion_m_train, accuracy_dev, confusion_m_dev, accuracy_test, confusion_m_test

def linearDiscriminantAnalysis(x_train, y_train, x_dev, y_dev, x_test, y_test):
	clf = LinearDiscriminantAnalysis()
	clf.fit(x_train, y_train)
	print("LINEAR DISCRIMINANT ANALYSIS")
	accuracy_train, confusion_m_train = evaluateClassificationModel(clf, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	accuracy_dev, confusion_m_dev = evaluateClassificationModel(clf, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	accuracy_test, confusion_m_test = evaluateClassificationModel(clf, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))

	return clf, accuracy_train, confusion_m_train, accuracy_dev, confusion_m_dev, accuracy_test, confusion_m_test

def knn(x_train, y_train, x_dev, y_dev, x_test, y_test, n_neighbors=20):
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(x_train, y_train)
	print("KNN")
	accuracy_train, confusion_m_train = evaluateClassificationModel(clf, x_train, y_train)
	print("training accuracy: " + str(accuracy_train))
	accuracy_dev, confusion_m_dev = evaluateClassificationModel(clf, x_dev, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	accuracy_test, confusion_m_test = evaluateClassificationModel(clf, x_test, y_test)
	print("test accuracy: " + str(accuracy_test))

	return clf, accuracy_train, confusion_m_train, accuracy_dev, confusion_m_dev, accuracy_test, confusion_m_test

def svm_pca(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'auto', kernel='rbf', C=.8, n_components=5):
	pca = PCA(n_components=n_components)
	x_train_new = pca.fit_transform(x_train)
	x_dev_new = pca.transform(x_dev)
	x_test_new = pca.transform(x_test)
	clf = SVC(gamma = gamma, kernel=kernel, C=C)
	clf.fit(x_train_new, y_train)
	print("SUPPORT VECTOR MACHINE WITH PCA")
	accuracy_train, confusion_m_train = evaluateClassificationModel(clf, x_train_new, y_train)
	print("training accuracy: " + str(accuracy_train))
	accuracy_dev, confusion_m_dev = evaluateClassificationModel(clf, x_dev_new, y_dev)
	print("dev accuracy: " + str(accuracy_dev))
	accuracy_test, confusion_m_test = evaluateClassificationModel(clf, x_test_new, y_test)
	print("test accuracy: " + str(accuracy_test))

	return clf, accuracy_train, confusion_m_train, accuracy_dev, confusion_m_dev, accuracy_test, confusion_m_test

def main(normalize=False, binary_encode=False, filter=False):
	non_categorical_columns = get_all_non_categorical()
	categorical_columns = get_all_categorical()
	x_train, y_train, x_dev, y_dev, x_test, y_test = load_and_clean(
		non_categorical_columns, categorical_columns, 
		normalize=normalize, binary_encode=binary_encode, filter=filter)
	# Run Models
	# REGRESSION
	linearRegression(x_train, y_train, x_dev, y_dev, x_test, y_test)
	linearRegression_L2(x_train, y_train, x_dev, y_dev, x_test, y_test, alpha=0.001)
	linearRegression_L1(x_train, y_train, x_dev, y_dev, x_test, y_test, alpha=0.005)
	supportVectorRegression(x_train, y_train, x_dev, y_dev, x_test, y_test, C=0.1, kernel='rbf', epsilon=0.1, shrinking=True) # TO DO: w One Hot Encoding, w no Normalization
	supportVectorRegression(x_train, y_train, x_dev, y_dev, x_test, y_test, C=0.1, kernel='rbf', epsilon=0.1, shrinking=True)
	supportVectorRegression(x_train, y_train, x_dev, y_dev, x_test, y_test, C=0.1, kernel='linear', epsilon=0.1, shrinking=True)

	# CLASSIFICATION
	k = 10
	y_train = convertToClass(y_train, k)
	y_dev = convertToClass(y_dev, k)
	y_test = convertToClass(y_test, k)
	# Run Models
	softmax(x_train, y_train, x_dev, y_dev, x_test, y_test, penalty='l1', C=0.8)
	svm(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'auto', kernel='rbf', C=1.5)
	svm(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'auto', kernel='linear', C=0.01)
	decisionTree(x_train, y_train, x_dev, y_dev, x_test, y_test, max_depth = 14)
	linearDiscriminantAnalysis(x_train, y_train, x_dev, y_dev, x_test, y_test)
	knn(x_train, y_train, x_dev, y_dev, x_test, y_test, n_neighbors=35)
	svm_pca(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'auto', kernel='rbf', C=1.0, n_components=5)

main(normalize=True, binary_encode=True, filter=True)