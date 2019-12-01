import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import category_encoders as ce
import pickle
from datetime import date


def remove_nan(X, non_c_features):
    for x in non_c_features:
        X[x] = X[x].fillna(X[x].mean())





def convert_data(data):
  
    if isinstance(data, (float, int)): #if the data already is number, we don't have to do anything
        return data
    data = data.replace(",","")
    data = data.replace("$", "")
    if data.endswith("%"):
        return float(data[0:-1])  / 100.0
    if data.endswith("T"):
        return float(data[0:-1]) * 10 ** 12
    if data.endswith("B"):
        return float(data[0:-1]) * 10 ** 9
    if data.endswith("M"):
        return float(data[0:-1]) * 10 ** 6
    if data.endswith("K"):
        return float(data[0:-1]) * 10 ** 3
    if data == "N/A":
        return "NaN"
    ret = None
    try:
        ret = float(data)
    except ValueError:
        ret = "NaN"
    return ret

def clean_data(X):
    return X.applymap(convert_data)


def google_trends_features():
    path1 = "../data/trends_IOT_all.pkl"
    path2 = "../data/trends_REG_all.pkl"

    d1, d2 = [pickle.load(open(p, "rb")) for p in [path1, path2]]

    avg5_us = d2.loc["United States"].to_numpy()

    split_size = 365  # average over this number of days

    start_index = d1.shape[0] - 2
    last_checkpoint = None
    current_sum = np.zeros(d1.shape[1])
    count = 0
    avg = []
    for i in range(start_index, 0, -2):
        if not last_checkpoint:
            last_checkpoint = d1.index[i].to_pydatetime()

        delta = d1.index[i].to_pydatetime() - last_checkpoint

        if abs(delta.days) >= split_size:
            avg.append(current_sum / count)
            current_sum = np.zeros(d1.shape[1])
            count = 0
            last_checkpoint = d1.index[i].to_pydatetime()
        else:
            current_sum += d1.iloc[i, :].to_numpy()
            count += 1

    result_dict = {"avg_{}".format(i): elem for i, elem in enumerate(avg)}

    result_dict["avg5_us"] = avg5_us

    return pd.DataFrame(result_dict)


Y = "Score"

def get_all_non_categorical():
    all =  ['year founded', 'current employee estimate', 'reviews', 'salaries', 'interviews','total employee estimate','Score', 'market_cap', 'enterprise_value', 'trailing_pe', 'forward_pe', 'peg_ratio_5', 'price_sales', 'price_book', 'enterprise_value_revenue', 'enterprise_value_ebitda', 'profit_margin', 'operating_margin', 'return_on_assets', 'return_on_equity', 'revenue',
                           'revenue_per_share', 'quarterly_revenue_share', 'gross_profit', 'ebitda', 'net_income_avi_to_common', 'diluted_eps', 'quarterly_earnings_growth', 'total_cash', 'total_cash_per_share', 'total_dept', 'total_dept_per_equity', 'operating_cash_flow', 'leveraged_free_cash_flow', 'stock_beta_3y', 'stock 52_week', 'stock_sp500_52_week', 'stock_52_week_high', 'stock_52_week_low', "website_rank", "organic_traffic", "traffic_cost", "linkedin_followers"]

    all.remove(Y)
    return all

def get_all_categorical():
    return ['industry', 'size range', 'city', ' state', 'country']


def load_and_clean(non_categorical, categorical, data_path="../data/with_stock_data_webclicks_linkedin.csv",
 normalize=False, binary_encode=False, trend_features = True, filter = False):

    frame = pd.read_csv(data_path)
    all_non_categorical = get_all_non_categorical()
    all_categorical = get_all_categorical()

    for non_c in non_categorical:
        if non_c not in all_non_categorical:
            raise ValueError("Non categorical {} is not valid".format(non_c))

    for c in categorical:
        if c not in all_categorical:
            raise ValueError("Categorical {} is not valid".format(non_c))

    X = pd.DataFrame({key: frame[key] for key in non_categorical})

    if trend_features:
        trends_data = google_trends_features()
        X = pd.concat([X, google_trends_features()], axis = 1)

    #X = pd.concat([X, ce_binary.fit_transform(X)])
    #y = frame[['Score']]
    y = frame[[Y]]
    for cat in categorical:
        new_cols = None
        if binary_encode:
            ce_binary = ce.BinaryEncoder(cols=[cat])
            new_cols = ce_binary.fit_transform(frame[cat])
        else:
            new_cols = pd.get_dummies(frame[cat], prefix='category')
        X = pd.concat([X, new_cols], axis=1)
    X = clean_data(X)
    remove_nan(X, non_categorical)


    if filter:
        indices = X["current employee estimate"] > 300
        X = X[indices]
        y = y[indices]
        
    print(y["Score"].value_counts())
    if (normalize):
        X.loc[:, :] = preprocessing.StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_test, y_test, test_size=.3, random_state=1)

    return (X_train, y_train, X_dev, y_dev, X_test, y_test)

def convert_to_class(y_data, k):
    eps = 0.001
    return y_data.apply(lambda x : np.floor(x / 5  * k - eps).astype(int))


if __name__ == "__main__":
    print(load_and_clean(["current employee estimate", "year founded"],[])[0].shape)
