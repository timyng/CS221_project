import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import random
from load_data import *
from sklearn.decomposition import PCA
import seaborn as sb
import matplotlib.colors as col


def convertToClass(Y, k):
    return Y.apply(lambda elem : np.round(elem * k / 5))


 
def plot_heat(X, y_data):
    K = 75
   
    #x_min, x_max, y_min, y_max = [X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()]
    m = 2.5
    x_min = np.mean(X[:,0]) - m * np.std(X[:,0])  
    x_max = np.mean(X[:,0]) + m * np.std(X[:,0])
    y_min = np.mean(X[:,1]) - m * np.std(X[:,1])
    y_max = np.mean(X[:,1]) + m * np.std(X[:,1])
    y, x = np.meshgrid(np.linspace(x_min, x_max, K), np.linspace(y_min, y_max, K))

    z = np.zeros([K, K])
    c = np.ones([K, K])

    width = x_max - x_min
    height = y_max - y_min
    dx = width / K
    dy = width / K
    for k, elem in enumerate(X):
        i = max(min(int((elem[0] - x_min) / width * K), K-1), 0)
        j = max(min(int((elem[1] - y_min) / width * K), K-1), 0)
        z[i][j] += y_data[k]
        c[i][j] += 1
    
    z = z / c
    print(z.shape)
    z = z[:-1, :-1]
    z_min, z_max = 1, 5

    fig, ax = plt.subplots()
    color_map = plt.get_cmap("hsv", 256)
    newcolors = color_map(np.linspace(0, 1, 256))
    newcolors[0,:] = (1,1,1, 1)
    new_color_map = col.ListedColormap(newcolors)
    c = ax.pcolormesh(x, y, z, vmin=z_min, cmap=new_color_map, vmax=z_max)

    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()
    plt.savefig("heatmap.png")


def save_scatter(X, y, name, ran = None):
    plt.figure()
    #plt.ylim(-1,1)
    colors = np.random.rand(y.shape[0])
    print(colors.shape)
    y_n = y.to_numpy().reshape(-1)
    color = y_n
    
    if ran:
        indices = (y_n >= ran[0]) & (y_n <= ran[1])
        color = 150 + (indices).astype(int) * 50
        plt.scatter(X[:,0][~indices], X[:,1][~indices], c = "blue", cmap=plt.cm.coolwarm)
        plt.scatter(X[:,0][indices], X[:,1][indices], c = "red", cmap=plt.cm.coolwarm)
    else:
        plt.scatter(X[:,0], X[:,1], c = color, cmap=plt.cm.coolwarm)
    plt.savefig("./{}.png".format(name))
    plt.show()
    plt.close()


def print_pca_info(pca, X):
    component_lengths = np.linalg.norm(pca.components_, axis=0)
    arg_sorted = np.argsort(component_lengths)[::-1]
    print(component_lengths[arg_sorted])
    print(X.columns[arg_sorted])

def remove_outliers(X, y, m = 2):

    y_n = y.to_numpy()
    print(abs(X[:,0] - np.mean(X[:,0])), np.std(X[:,0]))
    indices_0 = abs(X[:,0] - np.mean(X[:,0])) < m * np.std(X[:,0])
    indices_1 = abs(X[:,1] - np.mean(X[:,1])) < m * np.std(X[:,1])
    indices = indices_0 + indices_1
    return (X[indices], y.to_numpy()[indices]) 
    


def main():
    categorical = []
    #non_categorical = ["year "]
    non_categorical = ['year founded', 'current employee estimate', 'total employee estimate', 'reviews', 'salaries', 'interviews', 'market_cap', 'enterprise_value', 'trailing_pe', 'forward_pe', 'peg_ratio_5', 'price_sales', 'price_book', 'enterprise_value_revenue', 'enterprise_value_ebitda', 'profit_margin', 'operating_margin', 'return_on_assets', 'return_on_equity', 'revenue',
                           'revenue_per_share', 'quarterly_revenue_share', 'gross_profit', 'ebitda', 'net_income_avi_to_common', 'diluted_eps', 'quarterly_earnings_growth', 'total_cash', 'total_cash_per_share', 'total_dept', 'total_dept_per_equity', 'operating_cash_flow', 'leveraged_free_cash_flow', 'stock_beta_3y', 'stock 52_week', 'stock_sp500_52_week', 'stock_52_week_high', 'stock_52_week_low']
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_and_clean(
        non_categorical, categorical, normalize = True, binary_encode = True, trend_features = True)

    #print(X_train)
    pca = PCA(n_components=2)
    pca.fit(X_train)
    PCA_train_X = pca.transform(X_train)

    #X_f, y_f =  remove_outliers(PCA_train_X, y_train, m = 1)
    #print(X_f.shape, y_f.shape)

    plot_heat(PCA_train_X, y_train.to_numpy())

    print_pca_info(pca, X_train)
    
    return

    Y_class = convertToClass(y_train, 10)

    y_labels = ["{:.1f} to {:.1f}".format(y /2, (y+1)/2) for y in range(10)]
    
    for i,l in enumerate(y_labels):
        save_scatter(PCA_train_X, y_train, l, ran = (i/2, (i+1)/2))

    save_scatter(PCA_train_X, y_train, "PCA")


if __name__ == "__main__":
    main()