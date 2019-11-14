import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



data_path = "../data/with_stock_data.csv"
frame = pd.read_csv(data_path)

#keys = ["market_cap", "revenue", "total employee estimate", "enterprise_value", "stock 52_week", "leveraged_free_cash_flow"]
#keys = "market_cap,enterprise_value,trailing_pe,forward_pe,peg_ratio_5,price_sales,price_book,enterprise_value_revenue,enterprise_value_ebitda,profit_margin,operating_margin,return_on_assets,return_on_equity,revenue,revenue_per_share,quarterly_revenue_share,gross_profit,ebitda,net_income_avi_to_common,diluted_eps,quarterly_earnings_growth,total_cash,total_cash_per_share,total_dept,total_dept_per_equity,operating_cash_flow,leveraged_free_cash_flow,stock_beta_3y,stock 52_week,stock_sp500_52_week,stock_52_week_high,stock_52_week_low".split(",")
keys = ["industry"]

X = pd.DataFrame({key : frame[key] for key in keys})
print(X)

print(len(pd.unique(X["industry"])))

exit(0)

for x in X:
    X[x] = X[x].fillna(X[x].mean())



Y = pd.DataFrame({"y" : frame["Score"]})

X = pd.DataFrame({"rand" : np.random.random(Y.shape[0])})


reg = LinearRegression().fit(X, Y)



pred = reg.predict(X)



print(sum(np.abs(pred - Y.to_numpy()) <  0.1)/ Y.shape[0]) 

print("mean square error = {}".format(mean_squared_error(Y, pred)))


