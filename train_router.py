import time
from time import perf_counter
start_time=perf_counter()
from pandas import read_csv
from retrieval import model
model_load_time=perf_counter()
import numpy as np
import csv
def sigmoid_func(z):
    return 1/(1+np.exp(-np.clip(z,-400,400)))
queries=[]
labels=[]
with open("router_queries.csv", "r") as file:
    reader=csv.reader(file)
    header=next(reader)
    for row in reader:
        queries.append(row[0])
        labels.append(row[1])
labels=np.array(labels,dtype=np.int32)
queries=model.encode(queries)
weights=np.random.normal(0,0.01,queries.shape[1])
bias=0.0
iterations=500
lamda=0.1
learning_rate=0.2
model_train_start_time=perf_counter()
while(iterations):
    z=np.dot(queries,weights)+bias #forward pass starts
    predictions=sigmoid_func(z) #prediction for every query
    predictions+=1e-10
    predictions=np.clip(predictions,1e-15,1-1e-15)
    bce_loss=-np.mean(labels*np.log(predictions)+(1-labels)*np.log(1-predictions))
    l2_penalty=(lamda/(2*labels.size))*np.dot(weights,weights)
    loss=bce_loss+l2_penalty #forward pass ends
    dbce_dw=np.dot(queries.T,predictions-labels)/labels.size #backprop
    dl2_dw=lamda*weights/labels.size
    dL_dw=dbce_dw+dl2_dw
    dL_db=np.mean(predictions-labels)
    weights-=learning_rate*dL_dw
    bias-=learning_rate*dL_db
    iterations-=1
model_train_finish_time=perf_counter()
test_queries=[]
test_labels=[]
np.savez("router_weights.npz",w=weights,b=bias)

with open("router_test_queries.csv", "r") as file:
    reader=csv.reader(file)
    header=next(reader)
    for row in reader:
        test_queries.append(row[0])
        test_labels.append(row[1])
test_labels=np.array(test_labels,dtype=np.int32)
test_queries=model.encode(test_queries)
z=np.dot(test_queries,weights)+bias
output=sigmoid_func(z)
correct_or_not=(output>=0.5).astype(int)
correct=np.sum(correct_or_not==test_labels)
print("accuracy: ",correct/test_labels.size)
print(model_load_time-start_time,model_train_finish_time-model_train_start_time)










