# Get the stock symbol for company names

import csv
from tqdm import tqdm

def get_stock_symbol(company_name):
    database_filenames = ["NASDAQ.txt", "NYSE.txt", "AMEX.txt", "LSE.txt", \
        "OTCBB.txt", "TSX.txt", "SGX.txt"]
    targets = []
    for db_name in database_filenames:
        with open(db_name,'r') as f:
            targets = [line for line in f if company_name.lower() in line.lower()]
        if targets != []: return targets[0].split()[0]
    return []


csv_filename = "merged.csv"
csvfile = open(csv_filename)
reader = csv.reader(csvfile)
common_all = []
no_data = []
for row in tqdm(reader):
    temp = get_stock_symbol(row[1])
    if temp!=[]:
        common_all.append(row+[temp])
    else:
        no_data.append(row+[temp])
csvfile.close()

csvfile = open(csv_filename,'w')
for item in common_all:
    to_write = ','.join(item)
    print(to_write)
    csvfile.write(to_write)
csvfile.close()

output_name = "merged_with_stock_syms.csv"
f = open(output_name, "w")
for item in tqdm(common_all,desc="Writing to file"):
    to_write = ','.join(item)
    f.write(str(to_write)+"\n")
f.close()



# Get Google Trends data

import csv, time
from tqdm import tqdm
import pandas as pd
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
csv_data[:] = [item[1] for item in csv_data[1:]]
csv_file.close()

to_range = [500,505]
succeeded = []

for idx,company_name in tqdm(enumerate(csv_data[to_range[0]:to_range[1]])):
    kw_list = [company_name]
    pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
    trends_IOT_new = pytrends.interest_over_time()
    trends_REG_new = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)

    trends_IOT = pd.concat([trends_IOT,trends_IOT_new],axis=1) 
    trends_REG = pd.concat([trends_REG,trends_REG_new],axis=1) 
    del trends_IOT['isPartial']
    # del trends_REG['isPartial']
    succeeded.append(to_range[0]+idx)
    time.sleep(0.5)