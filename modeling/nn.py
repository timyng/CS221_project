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


class Net(nn.Module):
    def __init__(self, num_input, hidden, num_hidden_layers=2):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.fc1 = nn.Linear(num_input, hidden)
        self.fc_hidden = nn.Linear(hidden, hidden)
        self.fc_last = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = self.sig(self.fc1(x))  # First
        for i in range(self.num_hidden_layers):
            x = self.sig(self.fc_hidden(x))
       
        x = self.sig(self.fc_last(x))
        return x * 5

class Net2(nn.Module):
    def __init__(self, num_input, layers):
        super().__init__()
        
        layers = [num_input] + layers + [1]

        self.fc_layers = []
        for i in range(len(layers) - 1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.add_module("l_{}".format(i), self.fc_layers[-1])
        print(len(self.fc_layers))
    
    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return x 



#all features, 1 hidden, 100 => 0.3604
#all features, 1, 100, 25, 1 => 0.369
#all features, with trends, industry 1, 50, 25, 1 => 0.375
#all features, with trends, ["industry", " state", "size range"], 1,90,30,1 => 0.38


def accuracy(pred, truth, k = 0.25):
    return np.sum(np.abs(pred - truth) < k) / pred.shape[0]


def model_p_37_5():
    ModelParams = namedtuple("Model", ['non_categorical', 'categorical', 'binary_encode', 'use_trends', 'nn','nn_args', 'weight_decay', "lr", "name"])


    categorical = ["industry", " state", "size range"]
    non_categorical = ['year founded', 'current employee estimate', 'reviews', 'salaries', 'interviews','total employee estimate', 'market_cap', 'enterprise_value', 'trailing_pe', 'forward_pe', 'peg_ratio_5', 'price_sales', 'price_book', 'enterprise_value_revenue', 'enterprise_value_ebitda', 'profit_margin', 'operating_margin', 'return_on_assets', 'return_on_equity', 'revenue',
                           'revenue_per_share', 'quarterly_revenue_share', 'gross_profit', 'ebitda', 'net_income_avi_to_common', 'diluted_eps', 'quarterly_earnings_growth', 'total_cash', 'total_cash_per_share', 'total_dept', 'total_dept_per_equity', 'operating_cash_flow', 'leveraged_free_cash_flow', 'stock_beta_3y', 'stock 52_week', 'stock_sp500_52_week', 'stock_52_week_high', 'stock_52_week_low']
    
    net_args = [[90, 30]]
    return ModelParams(categorical = categorical ,
                    non_categorical = non_categorical,
                    binary_encode = True,
                    use_trends = True,
                    nn = Net2,
                    nn_args = net_args,
                    weight_decay = 0.03,
                    lr = 0.01,
                    name = "model_37_5")

def train_model_with_parapeters(model_p, save_stats  = True):

    training_epochs = 1000
    
    categorical = model_p.categorical
    non_categorical = model_p.non_categorical
    binary_encode = model_p.binary_encode
    use_trends = model_p.use_trends
    net = model_p.nn
    nn_args = model_p.nn_args
    weight_decay = model_p.weight_decay
    lr = model_p.lr
    

    X_train, y_train, X_dev, y_dev, X_test, y_test = load_and_clean(
        non_categorical, categorical, normalize = True, binary_encode = binary_encode, trend_features = use_trends)
    
    m = X_train.shape[1]

    model = net(m , *nn_args)
    
    X_var = Variable(torch.Tensor(X_train.to_numpy()))
    Y_var = Variable(torch.Tensor(y_train.to_numpy()))
    X_dev_var = Variable(torch.Tensor(X_dev.to_numpy()))

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay = weight_decay)


    learn_stats = {key : np.zeros(training_epochs) for key in ["train_loss", "train_accuracy", "dev_loss", "dev_accuracy"]}

    for i in range(training_epochs):
        y_pred = model(X_var)
        loss = loss_function(y_pred, Y_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
        if save_stats:
            y_pred_dev = model(X_dev_var).data.numpy()
            dev_acc = accuracy(y_pred_dev, y_dev)
            dev_loss = mean_squared_error(y_pred_dev, y_dev)
            train_acc = accuracy(y_pred.data.numpy(), y_train)

            learn_stats["train_loss"][i] = loss
            learn_stats["train_accuracy"][i] = train_acc
            learn_stats["dev_loss"][i] = dev_loss
            learn_stats["dev_accuracy"][i] = dev_acc
            print("Iteration = {} Accuracy  = {} Loss = {}".format(i, dev_acc, dev_loss))
        else:
            print("Iteration = {} Loss = {}".format(i, loss))
    

    x = np.linspace(0,training_epochs, training_epochs)
    plt.figure()
    for key in learn_stats:
        plt.plot(x, learn_stats[key], label = key)
    plt.ylim(0,.6)
    plt.legend()
    plt.title(model_p.name)
    plt.savefig("nn_dev_acc_loss.png")
    plt.show()



def main():
    torch.manual_seed(1)

    model_params = [model_p_37_5()]

    for model_p in model_params:
        train_model_with_parapeters(model_p)





if __name__ == "__main__":
    main()
