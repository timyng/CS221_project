import csv
from pprint import pprint
from difflib import SequenceMatcher

kaggle_path = "../../data/companies_sorted.csv"
glass_door_path = "../../data/glass_door.csv"


data = {}

first_kaggle = None #headers
first_glass = None #headers

i = 0
with open(kaggle_path) as csvfile:
    reader = csv.reader(csvfile)
    first_kaggle = next(reader)
    for row in reader:
        i+=1
        #if i > 100000:
        #    break
        if not row[1] in data:
            data[row[1]] = row



def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

result = {}



label_file = open(glass_door_path)
label_reader = csv.reader(label_file)
labels = [row for row in label_reader]

first_glass = labels[0]
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
    if s in data and not s in result:
        tmp = data[s] + label[2:]
        result[s] = tmp



print(result["google"])



def is_float(s):
    try:
        float(s)
    except ValueError:
        return False
    return True

keys_to_remove = []
for key in result.keys():
    fields_empty = False
    for elem in result[key]:
        if elem == '':
            fields_empty = True
            break
    if fields_empty:
        keys_to_remove.append(key)
        continue
    
    for i, elem in enumerate(result[key]):
        if is_float(elem[:-1]) and elem[-1] == "k":
                result[key][i] = float(elem[:-1]) * 1000 

for key in keys_to_remove:
    del result[key]         
        

outfile = open("out_merged.csv", "w")
writer = csv.writer(outfile)

writer.writerow(first_kaggle + first_glass[2:])
print(first_kaggle, first_glass)
for key in result.keys():
    #print("writing", result[key])
    writer.writerow(result[key])
