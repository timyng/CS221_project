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

def linearRegression_L1(x_train, y_train, x_test, y_test, alpha=0.1):
	reg = linear_model.Lasso(alpha=alpha)
	reg.fit(x_train, y_train)
	print("LINEAR REGRESSION L1")
	print("On Training Data:")
	mse, accuracy, confusion_m = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy))
	print("training MSE: " + str(mse))

	print("ON Test Data:")
	mse, accuracy, confusion_m = evaluateRegressionModel(reg, x_test, y_test)
	print("testing accuracy: " + str(accuracy))
	print("testing MSE: " + str(mse))

	return reg

def linearRegression_L2(x_train, y_train, x_test, y_test, alpha=1.0):
	reg = linear_model.Ridge(alpha=alpha)
	reg.fit(x_train, y_train)
	print("LINEAR REGRESSION L2")
	mse, accuracy, confusion_m = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy))
	print("training MSE: " + str(mse))

	print("ON Test Data:")
	mse, accuracy, confusion_m = evaluateRegressionModel(reg, x_test, y_test)
	print("testing accuracy: " + str(accuracy))
	print("testing MSE: " + str(mse))

	return reg

def linearRegression(x_train, y_train, x_test, y_test):
	reg = LinearRegression()
	reg.fit(x_train, y_train)
	print("LINEAR REGRESSION")
	print("On Training Data:")
	mse, accuracy, confusion_m = evaluateRegressionModel(reg, x_train, y_train)
	print("training accuracy: " + str(accuracy))
	print("training MSE: " + str(mse))

	print("On Test Data:")
	mse, accuracy, confusion_m = evaluateRegressionModel(reg, x_test, y_test)
	print("testing accuracy: " + str(accuracy))
	print("testing MSE: " + str(mse))

	return reg

def main(normalize=False, binary_encode=False, filter=False):
	non_categorical_columns = get_all_non_categorical()
	categorical_columns = get_all_categorical()
	x_train, y_train, x_test, y_test, x_test_unseen, y_test_unseen = load_and_clean(
		non_categorical_columns, categorical_columns, 
		normalize=normalize, binary_encode=binary_encode, filter=filter)
	# Run Models
	linearRegression(x_train, y_train, x_test, y_test)
	linearRegression_L2(x_train, y_train, x_test, y_test, alpha=1.0)
	reg = linearRegression_L1(x_train, y_train, x_test, y_test, alpha=0.1)
	coefficients = reg.coef_


main(normalize=False, binary_encode=False, filter=False)