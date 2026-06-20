import os
import numpy as np
weights_path="router_weights.npz"
if not os.path.exists(weights_path):
    raise FileNotFoundError("Weights file not found. Run train_router.py first")
data=np.load(weights_path)
weights=data["w"]
bias=data["b"]
def sigmoid_func(z):
    return 1/(1+np.exp(-np.clip(z,-400,400)))
def predict_cloud_prob(query_vec):
    z=np.dot(query_vec,weights)+bias
    return sigmoid_func(z)
from retrieval import get_query_vector,search_locally
q_vec=get_query_vector()
routing_decision=predict_cloud_prob(q_vec)
if(routing_decision>=0.5):
    print("\n--> [ROUTING] Escalating to cloud api (Complex Query)...")
else:
    print("\n--> [ROUTING] Processing locally...")
    max_sim,best_string,best_path=search_locally(q_vec)
    local_threshold=0.25
    if(max_sim<local_threshold):
        print(f"\n--> [ROUTING] Escalating to cloud api (Low similarity confidence: {max_sim:.4f})...")
    else:
        print("\n--- Best Local Match ---")
        print(f"File Source: {best_path}")
        print(f"Similarity Score: {max_sim:.4f}")
        print(f"Content: {best_string}\n")