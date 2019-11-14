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

