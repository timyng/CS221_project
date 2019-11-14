import matplotlib.pyplot as plt
import pandas as pd


data_path = "../data/with_stock_data.csv"
frame = pd.read_csv(data_path)


Y = pd.DataFrame({"y" : frame["Score"]})

print(Y)

plt.hist(Y.to_numpy(), bins =40)
plt.show()