import csv
from pprint import pprint
from difflib import SequenceMatcher

kaggle_path = "../../data/usa_companies.csv"
glass_door_path = "../../data/glass_door.csv"


data = {}


i = 0
with open(kaggle_path) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        i+=1
        #if i > 10000:
        #    break
        data[row[1]] = row



def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

result = {}


label_file = open(glass_door_path)
label_reader = csv.reader(label_file)
labels = [row for row in label_reader]
#    
#for elem in data.keys():
#    for label in labels:
#        sim_score = similar(label[0].strip().lower(), elem)
#        if sim_score > 0.95:
#            result[data[0], 0
#            print(elem, label, sim_score)
#
for label in labels:
    s = label[0].strip().lower()
    if label[0].strip().lower() in data:
        data[s] = data[s] + label[2:]
        result[s] = data[s]

for key in result.keys():
    print(result[key])


outfile = open("out_merged.csv", "w")
writer = csv.writer(outfile)

for key in result.keys():
    writer.writerow(result[key])
