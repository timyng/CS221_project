# Get Google Trends data

import csv
import time
from tqdm import tqdm
import pandas as pd
import pickle
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360)

# csvfile = open(csv_filename)
# reader = csv.reader(csvfile)
kw_list = ['dummy']
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
trends_IOT = pytrends.interest_over_time()
trends_REG = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)



# Read from csv and store header and data in memory.
csv_filename = "with_stock_data.csv"
csv_file = open(csv_filename)
csv_reader = csv.reader(csv_file, delimiter=',')
line_count = 0
csv_data = [item for item in csv_reader]
csv_file.close()
csv_header = csv_data[0]
csv_data[:] = [item[0] for item in csv_data[1:]]
csv_file.close()

to_range = [2000,3000]
succeeded = []

for idx,company_name in tqdm(enumerate(csv_data[to_range[0]:to_range[1]])):
    kw_list = [company_name]
    pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
    try:
        trends_IOT_new = pytrends.interest_over_time()
        trends_REG_new = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
    except:
        print("### Exited because Google rejected...")
        pickle.dump(trends_IOT, open("trends_IOT.p", "wb"))
        pickle.dump(trends_REG, open("trends_REG.p", "wb"))
        pickle.dump(succeeded, open("succeeded.p", "wb"))

    trends_IOT = pd.concat([trends_IOT,trends_IOT_new],axis=1) 
    trends_REG = pd.concat([trends_REG,trends_REG_new],axis=1) 
    if 'isPartial' in trends_IOT.keys(): del trends_IOT['isPartial']
    # del trends_REG['isPartial']
    succeeded.append(to_range[0]+idx)
    time.sleep(0.5)

pickle.dump(trends_IOT, open("trends_IOT.p", "wb"))
pickle.dump(trends_REG, open("trends_REG.p", "wb"))
pickle.dump(succeeded, open("succeeded.p", "wb"))
# ala = pickle.load(open("trends_IOT.p", "rb"))