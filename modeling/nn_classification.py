import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import random
from load_data import *
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from confusion import save_confusion_m
from sklearn.metrics import confusion_matrix
import os

OUT_BASE = "out/nn/"
class NN_classifier(nn.Module):
    def __init__(self, num_input, layers, num_output):
        super().__init__()
        
        layers = [num_input] + layers + [num_output]

        self.fc_layers = []
        for i in range(len(layers) - 1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.add_module("l_{}".format(i), self.fc_layers[-1])
        
        self.dropout = nn.Dropout(0.00)
        self.output_layer = nn.Softmax(dim=1)
        print(len(self.fc_layers))
    
    def forward(self, x):
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i in  [0,1]:
                x = self.dropout(x)
            x = F.relu(x)
    
        return self.output_layer(x)



class NN_2(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()

        self.fc1 = nn.Linear(num_input, 50)
        self.fc2 = nn.Linear(50, num_output)
        #self.fc3 = nn.Linear(25, num_output)
        self.training = True

    def forward(self, x):
        
        x = F.dropout(F.relu(self.fc1(x)),p = 0.3, training=self.training)
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

def weights_init_uniform(m):
    class_name = m.__class__.__name__
    if class_name.find("Linear") != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


class NN_3(nn.Module):
    def __init__(self, num_input):
        super().__init__()

        self.fc1 = nn.Linear(num_input, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)
        self.training = True

    def forward(self, x):
        
        x = F.dropout(F.relu(self.fc1(x)),p = 0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def weights_init_uniform(m):
    class_name = m.__class__.__name__
    if class_name.find("Linear") != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


def accuracy(pred, truth, k = 0.25):
    return np.sum(np.abs(pred.reshape(-1) - truth) < k) / pred.shape[0]

class NN_clf:
    def __init__(self, num_input, num_output, weight_decay, regression = False):
        self.regression = regression
        self.train_epochs = 600
        self.batch_size = 200
        self.model = None
        self.loss_function = None
        self.optimizer = None

        if regression:
            self.loss_function = nn.MSELoss()
            self.model = NN_3(num_input)
        else:
            self.loss_function = nn.CrossEntropyLoss()
            self.model = NN_2(num_input,num_output)

        self.model.apply(weights_init_uniform)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9, weight_decay = weight_decay)

    def fit(self, X, y, dev_X = None, dev_y = None):
        m = X.shape[1]
        n = X.shape[0]


        batch_data = []
        max_i = int(np.floor(n / self.batch_size) + 1)

        for i in range(max_i):
            i1 = i * self.batch_size
            i2 = (i + 1) * self.batch_size if i != max_i - 1 else n - 1
            if i2 - i1 > 50:
                batch_data.append((X[i1:i2], y[i1:i2]))


        X_dev_var = Variable(torch.Tensor(dev_X))
        X_train_var = Variable(torch.Tensor(X))
        accs = []
        for i in range(self.train_epochs):
            loss = None
            for batch_id, (x_batch, y_batch) in enumerate(batch_data):
                X_var = Variable(torch.Tensor(x_batch))
                Y_var = Variable(torch.Tensor(y_batch))
                if not self.regression:
                    Y_var = Y_var.long()
                y_pred = self.model(X_var)
                
                loss = self.loss_function(y_pred, Y_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if dev_X is not None and i % 20 == 0:
                dev_acc = self.score(dev_X, dev_y)
                train_acc = self.score(X, y)
                print("i = {} train accuracy = {:.3f}, dev accuracy = {:.3f}".format(i, train_acc, dev_acc))
                #if self.regression:
                #    print("MSE", self.MSE(dev_X, dev_y))
                accs.append((train_acc, dev_acc))

        return accs

    def predict(self, X):
        self.model.training = False
        X_var = Variable(torch.Tensor(X))
        result = None
        if self.regression:
            result = self.model(X_var).data.numpy()
        else:
            result =  np.apply_along_axis(np.argmax, 1, self.model(X_var).data.numpy())
        self.model.training = True
        return result

    def score(self, X, y):
        X_var = Variable(torch.Tensor(X))
        score = None
        pred = self.predict(X)
        if self.regression:
            score = accuracy(pred, y)
        else:
            score =  np.sum(y == pred)/y.shape[0]
        return score

    def MSE(self, X, y):
        self.training = False
        
        X_var = Variable(torch.Tensor(X))
        pred = self.model(X_var).data.numpy()
        print(pred.shape, y.shape)
        result =  mean_squared_error(pred, y)
        self.training  = True
        return result




def run_regression(train_X, train_y, dev_X, dev_y):
    weight_values = [0.04]
    scores = np.zeros_like(weight_values)
    for i,weight_value in enumerate(weight_values):
        clf = NN_clf(train_X.shape[1], 10,weight_value, regression=True)
        accs = clf.fit(train_X, train_y, dev_X, dev_y)
  
        y1, y2 = [np.array(arr) for arr in zip(*accs)]
        x = np.linspace(0, y1.shape[0] * 20, y1.shape[0])

        plt.figure()
        plt.plot(x, y1, label = "Train accuracy")
        plt.plot(x, y2, label = "Dev accuracy")
        plt.legend()
        plt.title("NN classifier weight_decay = {}".format(weight_value))
        plt.savefig("{}/nn_train_{}.png".format(OUT_BASE, weight_value))
        scores[i] = clf.score(dev_X, dev_y)
        plt.close()

def run_classification(train_X, train_y, dev_X, dev_y):
    weight_values = np.linspace(0.5,0, 20)
    scores = np.zeros((2, weight_values.shape[0]))
    for i,weight_value in enumerate(weight_values):
        clf = NN_clf(train_X.shape[1], 10,weight_value)
        accs = clf.fit(train_X, train_y, dev_X, dev_y)
  
        y1, y2 = [np.array(arr) for arr in zip(*accs)]
        x = np.linspace(0, y1.shape[0] * 20, y1.shape[0])

        plt.figure()
        plt.plot(x, y1, label = "Train accuracy")
        plt.plot(x, y2, label = "Dev accuracy")
        plt.legend()
        plt.title("NN classifier weight_decay = {}".format(weight_value))
        plt.savefig("{}/nn_train_{}.png".format(OUT_BASE, weight_value))
        scores[0][i] = clf.score(train_X, train_y)
        scores[1][i] = clf.score(dev_X, dev_y)
        #print("Result", scores[i], weight_value)
        plt.close()
        #y_pred = clf.predict(dev_X)
        #save_confusion_m(y_pred, dev_y,"Neural Net classifier {}".format(weight_value),y2[-1])
    return scores



def upsample_test(train_X, train_y, dev_X, dev_y, test_X, test_y):
    print(train_X.shape)
    train_X, train_y_class = upsample(train_X, train_y)
    print(train_X.shape)
    clf = NN_clf(train_X.shape[1], 10,0.01)
    clf.fit(train_X, train_y_class, dev_X, dev_y)

    print("test", clf.score( train_X, train_y_class))
    print("dev",clf.score( dev_X, dev_y))
    print("test", clf.score( test_X, test_y))

    y_pred = clf.predict(test_X)
    save_confusion_m(y_pred, test_y, "NN upsampled", clf.score( test_X, test_y))


def main():

    torch.manual_seed(0)

    categorical = ["industry"]
    non_categorical = ['year founded', 'current employee estimate', 'reviews', 'salaries', 'interviews','total employee estimate', 'market_cap', 'enterprise_value', 'trailing_pe', 'forward_pe', 'peg_ratio_5', 'price_sales', 'price_book', 'enterprise_value_revenue', 'enterprise_value_ebitda', 'profit_margin', 'operating_margin', 'return_on_assets', 'return_on_equity', 'revenue',
                           'revenue_per_share', 'quarterly_revenue_share', 'gross_profit', 'ebitda', 'net_income_avi_to_common', 'diluted_eps', 'quarterly_earnings_growth', 'total_cash', 'total_cash_per_share', 'total_dept', 'total_dept_per_equity', 'operating_cash_flow', 'leveraged_free_cash_flow', 'stock_beta_3y', 'stock 52_week', 'stock_sp500_52_week', 'stock_52_week_high', 'stock_52_week_low']

    train_X, train_y, dev_X, dev_y, test_X, test_y = load_and_clean(
        non_categorical, categorical, normalize = True, binary_encode = True, trend_features = True, filter = True)


    k = 10
    train_y_class = convert_to_class(train_y, k)
    dev_y_class = convert_to_class(dev_y, k)
    test_y_class = convert_to_class(test_y, k)

    dev_count = dev_y["Score"].value_counts().to_numpy()
    test_count = test_y["Score"].value_counts().to_numpy()
    print(dev_count[0] / np.sum(dev_count))

    train_X = train_X.to_numpy()
    train_y_class = train_y_class.to_numpy().flatten()
    dev_X = dev_X.to_numpy()
    dev_y_class = dev_y_class.to_numpy().flatten()
    test_X = test_X.to_numpy()
    test_y_class = test_y_class.to_numpy().flatten()

    train_y = train_y.to_numpy().flatten()
    dev_y = dev_y.to_numpy().flatten()
    test_y = test_y.to_numpy().flatten()

    

    #train_X, train_y_class = upsample(train_X, train_y_class)
    #res = run_classification(train_X, train_y_class, dev_X, dev_y_class)
    
    #upsample_test(train_X, train_y_class, dev_X, dev_y_class, test_X,test_y_class)
    #clf = NN_clf(train_X.shape[1], 10,0.0058)

    clf = NN_clf(train_X.shape[1], 10,0.058)

    #clf = NN_clf(train_X.shape[1], 10,0.01)

    clf.fit(train_X, train_y_class, dev_X, dev_y_class)
    print(clf.score(train_X, train_y_class))
    print(clf.score(dev_X, dev_y_class))
    print(clf.score(test_X, test_y_class))

    y_pred = clf.predict(test_X)
    save_confusion_m(y_pred , test_y_class, "NN", clf.score(test_X, test_y_class))

#
    #y_pred = clf.predict(test_X)
    #save_confusion_m(y_pred, test_y_class, "NN", clf.score( test_X, test_y))
    #run_regression(train_X, train_y, dev_X, dev_y)
    
    
    #x = np.linspace(0.2,0,20)
    #plt.plot(x,res[0],label =  "Train")
    #plt.plot(x, res[1], label = "Test")
    #plt.title("Adding regularization")
    #plt.xlabel("Weight_decay")
    #plt.ylabel("Accuracy")
    #plt.savefig("{}/weight_decay.png".format(OUT_BASE))
    #plt.show()





if __name__ == "__main__":
    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)

    main()