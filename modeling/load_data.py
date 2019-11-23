import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd



def remove_nan(X, non_c_features):
    for x in non_c_features:
        X[x] = X[x].fillna(X[x].mean())

def sig_inverse(y):
    return np.log(y / (1-y))



def load_and_clean(non_categorical, categorical, data_path = "../data/with_stock_data.csv"):
    
    frame = pd.read_csv(data_path)
    all_non_categorical = ['year founded', 'current employee estimate', 'total employee estimate', 'reviews', 'salaries', 'interviews', 'market_cap', 'enterprise_value', 'trailing_pe', 'forward_pe', 'peg_ratio_5', 'price_sales', 'price_book', 'enterprise_value_revenue', 'enterprise_value_ebitda', 'profit_margin', 'operating_margin', 'return_on_assets', 'return_on_equity', 'revenue', 'revenue_per_share', 'quarterly_revenue_share', 'gross_profit', 'ebitda', 'net_income_avi_to_common', 'diluted_eps', 'quarterly_earnings_growth', 'total_cash', 'total_cash_per_share', 'total_dept', 'total_dept_per_equity', 'operating_cash_flow', 'leveraged_free_cash_flow', 'stock_beta_3y', 'stock 52_week', 'stock_sp500_52_week', 'stock_52_week_high', 'stock_52_week_low']
    all_categorical = ['industry', 'size range', 'city', ' state', 'country']

    for non_c in non_categorical:
        if non_c not in all_non_categorical:
            raise ValueError("Non categorical {} is not valid".format(non_c))
    
    for c in categorical:
        if c not in all_categorical:
            raise ValueError("Categorical {} is not valid".format(non_c))

    
    X = pd.DataFrame({key : frame[key] for key in non_categorical})
    for cat in categorical:
        new_cols = pd.get_dummies(frame[cat], prefix = 'category')
        X = pd.concat([X, new_cols], axis = 1)
    remove_nan(X, non_categorical)
    
    y = frame[['Score']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=.3, random_state = 1)

    return (X_train, y_train, X_dev, y_dev, X_test, y_test)