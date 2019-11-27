# Scrape information about the company's domain

import csv
from tqdm import tqdm
from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup
import random
import string
import requests
import time

# driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
# content = driver.page_source
# time.sleep(5)
# scraping_domain = "https://www.similarweb.com/website/"
# driver.get(scraping_domain)

def get_website_clicks_v2(company_site_domain,driver,content):
    scraping_domain = "https://www.similarweb.com/website/" + company_site_domain

    driver.get(scraping_domain)
    content = driver.page_source

    # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko %s) Chrome/50.0.2661.102 Safari/537.36' % ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))} 

    # proxies = {'http': "http://116.71.132.53:8080", 'http': "http://79.120.177.106:8080", 'http': "http://207.144.111.230:8080", 'http': "http://87.250.109.174:8080", 'http': "http://194.44.246.82:8080", 'http': "http://51.158.113.142:8811", 'http': "http://51.68.141.240:3128", 'http': "http://190.152.12.54:41031", 'http': "http://181.115.168.69:49076", 'http': "http://37.98.175.251:81", 'http': "http://178.255.175.222:8080", 'http': "http://193.106.130.249:8080", 'http': "http://181.40.84.38:49674", 'http': "http://63.151.67.7:8080", 'http': "http://151.80.199.89:3128", 'http': "http://103.228.117.244:8080", 'http': "http://47.93.56.0:3128", 'http': "http://62.80.182.42:53281", 'http': "http://187.16.4.121:8080", 'http': "http://187.216.90.46:53281", 'http': "http://182.74.243.39:3128", 'http': "http://180.250.216.242:3128", 'http': "http://51.38.71.101:8080", 'http': "http://1.10.187.237:8080", 'http': "http://186.249.213.23:36586", 'http': "http://217.19.209.253:8080", 'http': "http://95.168.185.183:8080", 'http': "http://45.228.147.22:8080", 'http': "http://163.172.147.94:8811", 'http': "http://109.224.22.29:8080", 'http': "http://163.172.148.62:8811", 'http': "http://176.115.197.118:8080", 'http': "http://37.98.175.251:81", 'http': "http://51.158.111.242:8811", 'http': "http://180.183.243.4:8080", 'http': "http://192.119.203.170:48678", 'http': "http://1.10.187.237:8080", 'http': "http://41.78.243.233:53281", 'http': "http://51.158.108.135:8811", 'http': "http://188.235.138.182:34467", 'http': "http://202.142.158.114:8080", 'http': "http://183.82.116.56:8080", 'http': "http://46.151.145.4:53281", 'http': "http://94.75.76.10:8080", 'http': "http://176.227.188.66:53281", 'http': "http://182.48.87.170:8080", 'http': "http://144.217.74.219:3128", 'http': "http://190.248.153.162:8080", 'http': "http://217.168.76.230:59021", 'http': "http://178.134.152.46:41054", 'http': "http://31.131.67.14:8080", 'http': "http://45.76.43.163:8080", 'http': "http://112.14.47.6:52024", 'http': "http://51.158.106.54:8811", 'http': "http://112.14.47.6:52024", 'http': "http://79.120.177.106:8080", 'http': "http://148.217.94.54:3128", 'http': "http://91.234.127.222:53281", 'http': "http://87.250.109.174:8080", 'http': "http://119.18.147.111:8080", 'http': "http://86.110.27.165:3128", 'http': "http://213.109.234.4:8080", 'http': "http://84.22.198.26:8080", 'http': "http://192.119.203.124:48678", 'http': "http://51.158.123.35:8811", 'http': "http://198.11.178.14:8080", 'http': "http://183.88.212.141:8080", 'http': "http://122.50.5.147:10000", 'http': "http://41.206.57.242:8080", 'http': "http://82.81.169.142:80", 'http': "http://51.158.98.121:8811", 'http': "http://66.7.113.39:3128", 'http': "http://182.253.93.3:53281", 'http': "http://101.109.143.71:36127", 'http': "http://83.12.5.154:8080", 'http': "http://175.100.16.20:37725", 'http': "http://35.245.208.185:3128", 'http': "http://27.72.61.48:48455", 'http': "http://169.57.157.148:8123", 'http': "http://200.178.251.146:8080", 'http': "http://119.18.147.111:8080", 'http': "http://168.228.51.238:8080", 'http': "http://139.255.74.125:8080", 'http': "http://177.136.252.7:3128", 'http': "http://5.56.18.35:38827", 'http': "http://109.199.133.161:23500", 'http': "http://176.192.8.206:39422", 'http': "http://103.42.195.70:53281", 'http': "http://200.24.84.4:39136", 'http': "http://185.138.123.78:55630", 'http': "http://138.219.223.166:3128", 'http': "http://86.110.27.165:3128"}
    
    # r = requests.get(scraping_domain, headers = headers, proxies = proxies)
    soup = BeautifulSoup(content)

    temp = soup.findAll('', attrs={'class':'websiteRanks-valueContainer js-websiteRanksValue'})
    temp2 = soup.findAll('', attrs={'class':'engagementInfo-valueNumber js-countValue'})
    
    temp = list(temp)
    temp2 = list(temp2)

    global_rank = list(temp[0])[-1]
    country_rank = list(temp[1])[-1]

    total_visit = list(temp2[0])[-1]
    visit_duration = list(temp2[1])[-1]
    pages_per_visit = list(temp2[2])[-1]
    bounce_rate = list(temp2[3])[-1]

    for jj in [' ', ',', '\n']:
        global_rank = global_rank.replace(jj,'')
        country_rank = country_rank.replace(jj,'')

    time.sleep(5)
    return global_rank, country_rank, total_visit, visit_duration, pages_per_visit, bounce_rate


def get_website_clicks_v1(company_site_domain):
    scraping_domain = "https://www.similarweb.com/website/" + company_site_domain

    # driver.get(scraping_domain)

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko %s) Chrome/50.0.2661.102 Safari/537.36' % ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))} 

    proxies = {'http': "http://116.71.132.53:8080", 'http': "http://79.120.177.106:8080", 'http': "http://207.144.111.230:8080", 'http': "http://87.250.109.174:8080", 'http': "http://194.44.246.82:8080", 'http': "http://51.158.113.142:8811", 'http': "http://51.68.141.240:3128", 'http': "http://190.152.12.54:41031", 'http': "http://181.115.168.69:49076", 'http': "http://37.98.175.251:81", 'http': "http://178.255.175.222:8080", 'http': "http://193.106.130.249:8080", 'http': "http://181.40.84.38:49674", 'http': "http://63.151.67.7:8080", 'http': "http://151.80.199.89:3128", 'http': "http://103.228.117.244:8080", 'http': "http://47.93.56.0:3128", 'http': "http://62.80.182.42:53281", 'http': "http://187.16.4.121:8080", 'http': "http://187.216.90.46:53281", 'http': "http://182.74.243.39:3128", 'http': "http://180.250.216.242:3128", 'http': "http://51.38.71.101:8080", 'http': "http://1.10.187.237:8080", 'http': "http://186.249.213.23:36586", 'http': "http://217.19.209.253:8080", 'http': "http://95.168.185.183:8080", 'http': "http://45.228.147.22:8080", 'http': "http://163.172.147.94:8811", 'http': "http://109.224.22.29:8080", 'http': "http://163.172.148.62:8811", 'http': "http://176.115.197.118:8080", 'http': "http://37.98.175.251:81", 'http': "http://51.158.111.242:8811", 'http': "http://180.183.243.4:8080", 'http': "http://192.119.203.170:48678", 'http': "http://1.10.187.237:8080", 'http': "http://41.78.243.233:53281", 'http': "http://51.158.108.135:8811", 'http': "http://188.235.138.182:34467", 'http': "http://202.142.158.114:8080", 'http': "http://183.82.116.56:8080", 'http': "http://46.151.145.4:53281", 'http': "http://94.75.76.10:8080", 'http': "http://176.227.188.66:53281", 'http': "http://182.48.87.170:8080", 'http': "http://144.217.74.219:3128", 'http': "http://190.248.153.162:8080", 'http': "http://217.168.76.230:59021", 'http': "http://178.134.152.46:41054", 'http': "http://31.131.67.14:8080", 'http': "http://45.76.43.163:8080", 'http': "http://112.14.47.6:52024", 'http': "http://51.158.106.54:8811", 'http': "http://112.14.47.6:52024", 'http': "http://79.120.177.106:8080", 'http': "http://148.217.94.54:3128", 'http': "http://91.234.127.222:53281", 'http': "http://87.250.109.174:8080", 'http': "http://119.18.147.111:8080", 'http': "http://86.110.27.165:3128", 'http': "http://213.109.234.4:8080", 'http': "http://84.22.198.26:8080", 'http': "http://192.119.203.124:48678", 'http': "http://51.158.123.35:8811", 'http': "http://198.11.178.14:8080", 'http': "http://183.88.212.141:8080", 'http': "http://122.50.5.147:10000", 'http': "http://41.206.57.242:8080", 'http': "http://82.81.169.142:80", 'http': "http://51.158.98.121:8811", 'http': "http://66.7.113.39:3128", 'http': "http://182.253.93.3:53281", 'http': "http://101.109.143.71:36127", 'http': "http://83.12.5.154:8080", 'http': "http://175.100.16.20:37725", 'http': "http://35.245.208.185:3128", 'http': "http://27.72.61.48:48455", 'http': "http://169.57.157.148:8123", 'http': "http://200.178.251.146:8080", 'http': "http://119.18.147.111:8080", 'http': "http://168.228.51.238:8080", 'http': "http://139.255.74.125:8080", 'http': "http://177.136.252.7:3128", 'http': "http://5.56.18.35:38827", 'http': "http://109.199.133.161:23500", 'http': "http://176.192.8.206:39422", 'http': "http://103.42.195.70:53281", 'http': "http://200.24.84.4:39136", 'http': "http://185.138.123.78:55630", 'http': "http://138.219.223.166:3128", 'http': "http://86.110.27.165:3128"}
    
    r = requests.get(scraping_domain, headers = headers, proxies = proxies)
    soup = BeautifulSoup(r.text, "html.parser")

    temp = soup.findAll('', attrs={'class':'websiteRanks-valueContainer js-websiteRanksValue'})
    temp2 = soup.findAll('', attrs={'class':'engagementInfo-valueNumber js-countValue'})
    
    temp = list(temp)
    temp2 = list(temp2)

    global_rank = list(temp[0])[-1]
    country_rank = list(temp[1])[-1]

    total_visit = list(temp2[0])[-1]
    visit_duration = list(temp2[1])[-1]
    pages_per_visit = list(temp2[2])[-1]
    bounce_rate = list(temp2[3])[-1]

    for jj in [' ', ',', '\n']:
        global_rank = global_rank.replace(jj,'')
        country_rank = country_rank.replace(jj,'')
    return global_rank, country_rank, total_visit, visit_duration, pages_per_visit, bounce_rate


csv_filename = "./with_stock_data.csv"
csvfile = open(csv_filename)
reader = csv.reader(csvfile)
common_all = []

for idx,row in tqdm(enumerate(reader)):
    if idx == 0:
        common_all.append(row+['global_rank', 'country_rank', 'total_visit', 'visit_duration', 'pages_per_visit', 'bounce_rate'])
    else:
        company_site_domain = row[1]
        global_rank, country_rank, total_visit, visit_duration, pages_per_visit, bounce_rate = get_website_clicks_v1(company_site_domain)
        common_all.append(row+[global_rank, country_rank, total_visit, visit_duration, pages_per_visit, bounce_rate])
        print(company_site_domain, global_rank, total_visit)
        time.sleep(60)

csvfile.close()


csv_filename = "../data/with_stock_data.csv"
csvfile = open(csv_filename)
reader = csv.reader(csvfile)
company_all_domains = []
for idx,row in tqdm(enumerate(reader)):
    if idx==0:
        continue
    else:
        company_all_domains.append(row[1])
csvfile.close()



# def get_stock_symbol(company_name):
#     database_filenames = ["NASDAQ.txt", "NYSE.txt", "AMEX.txt", "LSE.txt", \
#         "OTCBB.txt", "TSX.txt", "SGX.txt"]
#     targets = []
#     for db_name in database_filenames:
#         with open(db_name,'r') as f:
#             targets = [line for line in f if company_name.lower() in line.lower()]
#         if targets != []: return targets[0].split()[0]
#     return []


csv_filename = "with_stock_data.csv"
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

output_name = "with_stock_data_webclicks.csv"
f = open(output_name, "w")
for item in tqdm(common_all,desc="Writing to file"):
    to_write = ','.join(item)
    f.write(str(to_write)+"\n")
f.close()

