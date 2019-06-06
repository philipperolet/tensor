import torch
import csv
import numpy as np

res = dict()
with open('/home/filou/Downloads/activ.csv') as csv_file:
    rows = csv.reader(csv_file, delimiter=',')
    next(rows, None)
    for row in rows:
        if ("Waiting" in row[5]
            or not(row[3])
            or row[3] == "0"):
            continue
        win = 1 if "Won" in row[5] else 0
        criterion = row[0]
        if not(criterion in res):
            res[criterion] = list()
        res[criterion].append([(float(row[3])-1.0)/4, win])

print(res)
res['test 1'] = [ [1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [0, 0], [0, 0]]
res['test 2'] = [ [1, 0], [0, 1], [1, 0], [0, 1] ]
res['test 3'] = [ [1, 1], [0.75, 1], [0.75, 1], [1, 1], [1, 0], [0.75, 0]]
sorted_res = []
for crit, data in res.items():
    values = np.array(data)
    covariance = np.cov(values[:, 0], values[:, 0])[0][1]
    correlation = np.corrcoef(values[:, 0], values[:, 1])[0][1]
    corr_bis = covariance / np.sqrt(np.var(values[:, 0]) * np.var(values[:, 1]))
    sorted_res.append([correlation, crit])
    print(f"{correlation:.2}, {crit}")

for item in sorted(sorted_res, key=lambda x: x[0], reverse=True):
    # print(f"{item[0]:.2}")
    print(f"{item[1]}")
    # print(f"{item[0]:.2}, {item[1]}")
    
