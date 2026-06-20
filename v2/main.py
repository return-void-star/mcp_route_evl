from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
from database import init_db,get_conn,DB_PATH
import os
import numpy as np

if not os.path.isfile(DB_PATH):
    init_db()

from indexer import run_indexer
indexing_needed=1
if(indexing_needed):
    run_indexer(model,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data"))

from train_router import train_and_save_neuron,testing_show
training_needed=1
test_show=0
neuron_path="router_config.npz"
if(training_needed):
    train_and_save_neuron(neuron_path,model,500,0.1,0.3)
if(test_show):
    testing_show(neuron_path,model)
if not os.path.exists(neuron_path):
    raise FileNotFoundError("Weights file not found. Run train_router.py first")
n_data= np.load(neuron_path)
weights = n_data["w"]
bias = n_data["b"]

from router import predict_cloud_prob
from retrieval import search_locally
def prompt_query():
    return input("Enter ur query: ")
query_string=prompt_query()
query_vec=model.encode(query_string)
escalate_prob=predict_cloud_prob(query_vec,weights,bias)
if(escalate_prob>=0.5):
    print("\n--> [ROUTING] Escalating to cloud api (Complex Query)...")
else:
    print("\n--> [ROUTING] Processing locally...")
    max_sim, best_string, best_path = search_locally(query_vec)
    local_threshold = 0.25
    if (max_sim < local_threshold):
        print(f"\n--> [ROUTING] Escalating to cloud api (Low similarity confidence: {max_sim:.4f})...")
    else:
        print("\n--- Best Local Match ---")
        print(f"File Source: {best_path}")
        print(f"Similarity Score: {max_sim:.4f}")
        print(f"Content: {best_string}\n")

