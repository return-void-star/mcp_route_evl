import numpy as np
def sigmoid_func(z):
    return 1/(1+np.exp(-np.clip(z,-400,400)))
def predict_cloud_prob(query_vec,weights,bias):
    z=np.dot(query_vec,weights)+bias
    return sigmoid_func(z)
