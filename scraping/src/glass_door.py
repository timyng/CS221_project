import requests
import random
import string
import time
import csv
from bs4 import BeautifulSoup


N = 4
N_start = 0
global_soup = None
global_alt = None
fails = 0

with open("output.csv", "wb") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Name", "Location", "Score", "reviews", "salaries", "interviews"])
    for i in range(N_start, N_start + N):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko %s) Chrome/50.0.2661.102 Safari/537.36' %
        ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))}
        url = "https://www.glassdoor.com/Reviews/us-reviews-SRCH_IL.0,2_IN1_IP%s.htm" % i
        time.sleep(1)
        print("Now downloading %d to %d" %(i*10 + 1, (i+1) * 10))
        print(url)
        r = requests.get(url, headers = headers)
        soup = BeautifulSoup(r.text, "html.parser")
        names = [elem.text.encode('ascii', 'ignore') for elem in soup.findAll("a", {"class":"tightAll h2"})]
        locations = [elem.text.encode("ascii", "ignore") for elem in soup.findAll("span", {"class", "hqInfo adr"})]
        numbers = [elem.text.encode("ascii", "ignore") for elem in soup.findAll("span", {"class", "num h2"})]

        if len(names) == 0: #if they send a different variant of the page
            print("alt")
            global_alt = soup
            locations = [elem.text.encode("ascii", "ignore") for elem in soup.findAll("p", {"class", "hqInfo adr m-0"})]
            names = [elem["alt"].split()[0].encode('ascii', 'ignore') for elem in soup.findAll("img", {"class",""})]
        else:
            global_soup = soup
        scores = [elem.text.encode('ascii', 'ignore') for elem  in soup.findAll("span", {"class":"bigRating strong margRtSm h2"})]
        if len(names) == 0:
            
            if (fails > 10):
                break
            print("something went wrong!")
            print(r.text)
            global_soup = soup
            print("try again")
            i-=1
            continue
            fails += 1
        fails = 0

        reviews = [numbers[i*3] for i in range(len(numbers)/3)]
        salaries = [numbers[i*3 + 1] for i in range(len(numbers)/3)]
        interviews = [numbers[i*3 + 2] for i in range(len(numbers)/3)]
        for row in zip(names,locations, scores, reviews, salaries, interviews):
            writer.writerow([e.strip() for e in row])