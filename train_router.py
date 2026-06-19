from retrieval import model
import numpy as np
import csv
queries=[]
labels=[]
with open("data/router_queries.csv","r") as file:
    reader=csv.reader(file)
    header=next(reader)
    for row in reader:
        queries.append(row[0])
        labels.append(row[1])
labels=np.array(labels,dtype=np.int32)
queries=model.encode(queries)
print(labels.shape,queries.shape)
weights=np.random.normal(0,0.01,labels.size)
