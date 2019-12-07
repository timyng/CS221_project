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


def main():

    torch.manual_seed(0)

    categorical = ["industry"]
    non_categorical = ['year founded', 'current employee estimate', 'reviews', 'salaries', 'interviews','total employee estimate', 'market_cap', 'enterprise_value', 'trailing_pe', 'forward_pe', 'peg_ratio_5', 'price_sales', 'price_book', 'enterprise_value_revenue', 'enterprise_value_ebitda', 'profit_margin', 'operating_margin', 'return_on_assets', 'return_on_equity', 'revenue',
                           'revenue_per_share', 'quarterly_revenue_share', 'gross_profit', 'ebitda', 'net_income_avi_to_common', 'diluted_eps', 'quarterly_earnings_growth', 'total_cash', 'total_cash_per_share', 'total_dept', 'total_dept_per_equity', 'operating_cash_flow', 'leveraged_free_cash_flow', 'stock_beta_3y', 'stock 52_week', 'stock_sp500_52_week', 'stock_52_week_high', 'stock_52_week_low']

    train_X, train_y, dev_X, dev_y, test_X, test_y = load_and_clean(
        non_categorical, categorical, normalize = True, binary_encode = True, trend_features = True, filter = True)


    k = 10
    train_y = convert_to_class(train_y, k)
    dev_y = convert_to_class(dev_y, k)
    test_y = convert_to_class(test_y, k)

    dev_count = dev_y["Score"].value_counts().to_numpy()
    test_count = test_y["Score"].value_counts().to_numpy()
    print(dev_count[0] / np.sum(dev_count))
    print(test_count[0] / np.sum(test_count))

    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy().flatten()
    dev_X = dev_X.to_numpy()
    dev_y = dev_y.to_numpy().flatten()
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy().flatten()

    training_epochs = 600

    batch_size = 60
    

    m = train_X.shape[1]
    n = train_X.shape[0]
    layers = [70,20]

    model = NN_classifier(m , layers, k)
    

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay = 0.008)


    batch_data = []
    max_i = int(np.floor(n / batch_size) + 1)
   
    for i in range(max_i):
        i1 = i * batch_size
        i2 = (i + 1) * batch_size if i != max_i - 1 else n - 1
        if i2 - i1 > 50:
            batch_data.append((train_X[i1:i2], train_y[i1:i2]))

    X_dev_var = Variable(torch.Tensor(dev_X))
    X_train_var = Variable(torch.Tensor(train_X))
    X_test_var = Variable(torch.Tensor(test_X))
    accs = []
    
    for i in range(training_epochs):
        loss = None
        for batch_id, (x_batch, y_batch) in enumerate(batch_data):
            X_var = Variable(torch.Tensor(x_batch))
            Y_var = Variable(torch.Tensor(y_batch)).long()
            y_pred = model(X_var)
            loss = loss_function(y_pred, Y_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 20 == 0:
            dev_acc = np.sum(dev_y == np.apply_along_axis(np.argmax, 1, model(X_dev_var).data.numpy()))/dev_y.shape[0]
            train_acc = np.sum(train_y == np.apply_along_axis(np.argmax, 1, model(X_train_var).data.numpy()))/train_y.shape[0]
            test_acc = np.sum(test_y == np.apply_along_axis(np.argmax, 1, model(X_test_var).data.numpy()))/test_y.shape[0]
            print("i = {} train accuracy = {:.3f}, dev accuracy = {:.3f}".format(i, train_acc, dev_acc))
            accs.append((train_acc, dev_acc, test_acc))
    

    y1, y2, y3 = [np.array(arr) for arr in zip(*accs)]
    x = np.linspace(0, y1.shape[0] * 20, y1.shape[0])

    plt.figure()
    plt.plot(x, y1, label = "Train accuracy")
    plt.plot(x, y2, label = "Test accuracy")
    #plt.plot(x, y3, label = "Test accuracy")
    plt.legend()
    plt.title("NN classifier")
    plt.savefig("{}/nn_train.png".format(OUT_BASE))
    plt.show()

    
    y_pred = np.apply_along_axis(np.argmax, 1, model(X_dev_var).data.numpy())

    save_confusion_m(y_pred, dev_y,"Neural Net classifier",y2[-1])
    print(m)


if __name__ == "__main__":
    if not os.path.exists(OUT_BASE):
        os.makedirs(OUT_BASE)

    main()