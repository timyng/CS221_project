import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from traditional_models import *

def decisionTreeRegressor_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test, max_depth=None):
	param_list = []
	for i in range(1, 100):
		param_list.append(i)
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = decisionTreeRegressor(x_train, y_train, x_dev, y_dev, x_test, y_test, max_depth=param)
		accuracy_dev = results[5]
		accuracy_train = results[2]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("Decision Tree Regressor: Max Depth vs. Accuracy")
	plt.xlabel("Max Depth")
	plt.savefig("DecisionTreeDepth_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

# def svm_reg_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
# 	param_list = [0.01, 0.1, 0.2, 0.3, 0.5, .7, 0.75, 0.8, 0.85, .9, 0.95, 1, 1.5, 2]
# 	dev_accuracy_list = []
# 	train_accuracy_list = []
# 	for param in param_list:
# 		print("--------------------------------------------------")
# 		print(param)
# 		results = supportVectorRegression(x_train, y_train, x_dev, y_dev, x_test, y_test, C=param, kernel='rbf', epsilon=0.1, shrinking=True)
# 		accuracy_dev = results[5]
# 		accuracy_train = results[2]
# 		dev_accuracy_list.append(accuracy_dev)
# 		train_accuracy_list.append(accuracy_train)
# 		print("--------------------------------------------------")
# 	plt.figure()
# 	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
# 	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
# 	plt.legend()
# 	#plt.ylim(0, 0.6)
# 	plt.title("Support Vector Regression: Penalty Parameter vs. Accuracy")
# 	plt.xlabel("Penalty parameter C")
# 	plt.savefig("SupportVectorRegression_C_vs_Accuracy", bbox_inches = 'tight')
# 	plt.show()

def linear_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test, alpha=0.1):
	param_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.1, 0.2, 0.3, 0.5, .7, 0.75, 0.8, 0.85, .9, 0.95, 1, 1.2, 1.4, 1.6, 2, 2.5, 3, 4, 5, 6, 7, 8, 9]
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = linearRegression_L2(x_train, y_train, x_dev, y_dev, x_test, y_test, alpha=param)
		accuracy_dev = results[5]
		accuracy_train = results[2]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("Linear Regression with L2 reg: Alpha vs. Accuracy")
	plt.xlabel("Alpha")
	plt.savefig("LinearRegressionAlpha_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def svm_reg_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	# param_list = [0.01, 0.1, 0.2, 0.3, 0.5, .7, 0.75, 0.8, 0.85, .9, 0.95, 1, 1.5, 2]
	param_list = ["linear", "poly", "rbf", "sigmoid"]
	dev_accuracy_list = []
	train_accuracy_list = []
	test_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = supportVectorRegression(x_train, y_train, x_dev, y_dev, x_test, y_test, C=0.8, kernel=param, epsilon=0.1, shrinking=True)
		accuracy_dev = results[5]
		accuracy_train = results[2]
		accuracy_test = results[8]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		test_accuracy_list.append(accuracy_test)
		print("--------------------------------------------------")
	plt.figure()
	# plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	# plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.bar(param_list, test_accuracy_list, label = "Test Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("Support Vector Regression: Kernels vs. Accuracy")
	plt.xlabel("Kernel")
	plt.savefig("SupportVectorRegression_Kernels_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def svm_reg_pca_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	param_list = []
	for i in range(1, 91):
		param_list.append(i)
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = supportVectorRegressionPCA(x_train, y_train, x_dev, y_dev, x_test, y_test, C=0.8, kernel='rbf', epsilon=0.1, shrinking=True, n_components=param)
		accuracy_dev = results[5]
		accuracy_train = results[2]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("Support Vector Regression with PCA: # of Components vs. Accuracy")
	plt.xlabel("# of Components")
	plt.savefig("SupportVectorRegression_Components_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def softmax_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	param_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.5, .7, 0.75, 0.8, 0.85, .9, 0.95, 1]
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = softmax(x_train, y_train, x_dev, y_dev, x_test, y_test, penalty='l2', C=param)
		accuracy_dev = results[3]
		accuracy_train = results[1]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("Softmax Classification: C vs. Accuracy")
	plt.xlabel("C")
	plt.savefig("Softmax_C_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def svm_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	param_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.5, .7, 0.75, 0.8, 0.85, .9, 0.95, 1]
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = svm(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'auto', kernel='rbf', C=param)
		accuracy_dev = results[3]
		accuracy_train = results[1]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("SVM Classification: C vs. Accuracy")
	plt.xlabel("C")
	plt.savefig("SVM_C_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def decisionTree_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	param_list = []
	for i in range(1, 40):
		param_list.append(i)
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = decisionTree(x_train, y_train, x_dev, y_dev, x_test, y_test, max_depth = param)
		accuracy_dev = results[3]
		accuracy_train = results[1]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("Decision Tree: Max Depth vs. Accuracy")
	plt.xlabel("Max Depth")
	plt.savefig("DecisionTree_Depth_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def knn_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	param_list = []
	for i in range(1, 40):
		param_list.append(i)
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = knn(x_train, y_train, x_dev, y_dev, x_test, y_test, n_neighbors=param)
		accuracy_dev = results[3]
		accuracy_train = results[1]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("KNN: Nearest Neighbors vs. Accuracy")
	plt.xlabel("# of Nearest Neighbors")
	plt.savefig("NearestNeighbors_K_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def svm_pca_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	param_list = []
	for i in range(1, 91):
		param_list.append(i)
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = svm_pca(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'auto', kernel='rbf', C=.95, n_components=param)
		accuracy_dev = results[3]
		accuracy_train = results[1]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("SVM Classification with PCA: # of Components  vs. Accuracy")
	plt.xlabel("# of Components")
	plt.savefig("SVM_Components_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def lgb_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test):
	# param_list = [5, 10, 15, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	param_list = []
	for i in range(2, 100):
		param_list.append(i)
	dev_accuracy_list = []
	train_accuracy_list = []
	for param in param_list:
		print("--------------------------------------------------")
		print(param)
		results = lightgradientboosting(x_train, y_train, x_dev, y_dev, x_test, y_test, n_estimators=50, application='multiclass', lambda_l2=10.0, bagging_fraction=0.6, bagging_freq=1, num_class=10, max_depth=2, num_leaves = param)
		accuracy_dev = results[3]
		accuracy_train = results[1]
		dev_accuracy_list.append(accuracy_dev)
		train_accuracy_list.append(accuracy_train)
		print("--------------------------------------------------")
	plt.figure()
	plt.plot(param_list, dev_accuracy_list, label = "Dev Accuracy")
	plt.plot(param_list, train_accuracy_list, label = "Train Accuracy")
	plt.legend()
	#plt.ylim(0, 0.6)
	plt.title("Light Gradient Boosting: Number of Leaves vs. Accuracy")
	plt.xlabel("# of Leaves")
	plt.savefig("LGB_Leaves_vs_Accuracy", bbox_inches = 'tight')
	plt.show()

def main(normalize=False, binary_encode=False, filter=False):
	non_categorical_columns = get_all_non_categorical()
	categorical_columns = get_all_categorical()
	x_train, y_train, x_dev, y_dev, x_test, y_test = load_and_clean(
		non_categorical_columns, categorical_columns, 
		normalize=normalize, binary_encode=binary_encode, filter=filter)
	
	# FINETUNING
	# svm_reg_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# svm_reg_pca_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# linear_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# decisionTreeRegressor_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)

	# REGRESSION
	# BASELINE REGRESSION MODELS
	# linearRegression(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# supportVectorRegression(x_train, y_train, x_dev, y_dev, x_test, y_test, C=1.0, kernel='rbf', epsilon=0.1, shrinking=True)
		
	# CLASSIFICATION -----------------
	k = 10
	y_train = convertToClass(y_train, k)
	y_dev = convertToClass(y_dev, k)
	y_test = convertToClass(y_test, k)

	# FINETUNING
	# softmax_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# svm_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# decisionTree_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# knn_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# svm_pca_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# lgb_parameter_tuning(x_train, y_train, x_dev, y_dev, x_test, y_test)

	# BASELINE CLASSIFICATION MODELS
	# softmax(x_train, y_train, x_dev, y_dev, x_test, y_test, penalty='l2', C=1.0)
	# svm(x_train, y_train, x_dev, y_dev, x_test, y_test, gamma = 'scale', kernel='rbf', C=1.0)
	# decisionTree(x_train, y_train, x_dev, y_dev, x_test, y_test, max_depth = None)
	# linearDiscriminantAnalysis(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# quadraticDiscriminantAnalysis(x_train, y_train, x_dev, y_dev, x_test, y_test)
	# knn(x_train, y_train, x_dev, y_dev, x_test, y_test, n_neighbors=5)

main(normalize=True, binary_encode=True, filter=True)