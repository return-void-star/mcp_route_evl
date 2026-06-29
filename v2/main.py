import os

from torch.package import ObjMismatchError
from watchdog.observers import Observer

from download_onnx import ONNXEmbedder

model=ONNXEmbedder()

from database import init_db,DB_PATH

import numpy as np

if not os.path.isfile(DB_PATH):
    init_db()

from file_watcher import start_file_watcher
watcher=start_file_watcher(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data"),model)


'''
from indexer import run_indexer
indexing_needed=1
if(indexing_needed):
    run_indexer(model,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data"))
'''
from train_router import train_and_save_neuron,testing_show
training_needed=0
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

'''
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
        print(f"Content: {best_string}\n")'''

from router import predict_cloud_prob
from retrieval import search_locally
from PySide6.QtWidgets import QApplication
from gui import SearchWidget

def search_engine_callback(query):
    query_vec=model.encode(query)
    escalation_prob=predict_cloud_prob(query_vec,weights,bias)

    if(escalation_prob>=0.5):
        return """
                <div style='background-color: rgba(10, 132, 255, 0.12); border: 1px solid rgba(10, 132, 255, 0.25); border-radius: 8px; padding: 14px;'>
                  <div style='color: #0A84FF; font-weight: 600; font-size: 14px; font-family: .AppleSystemUIFont, sans-serif;'>☁️ Escalating to Cloud AI</div>
                  <div style='color: #CCCCCC; font-size: 13px; font-family: .AppleSystemUIFont, sans-serif; margin-top: 6px; line-height: 1.4;'>
                    This is a complex query requiring synthetic reasoning. Routing to cloud model...
                  </div>
                </div>
                """
    max_sim,best_string,best_path=search_locally(query_vec)
    local_threshold=0.20

    if(max_sim<local_threshold):
        return f"""
                <div style='background-color: rgba(255, 69, 58, 0.12); border: 1px solid rgba(255, 69, 58, 0.25); border-radius: 8px; padding: 14px;'>
                  <div style='color: #FF453A; font-weight: 600; font-size: 14px; font-family: .AppleSystemUIFont, sans-serif;'>⚠️ Low Confidence Local Search ({max_sim:.4f})</div>
                  <div style='color: #CCCCCC; font-size: 13px; font-family: .AppleSystemUIFont, sans-serif; margin-top: 6px; line-height: 1.4;'>
                    No matching local knowledge was found in the database. Escalating to cloud model...
                  </div>
                </div>
                """

    file_name=os.path.basename(best_path) if best_path else "Unknown File"
    clean_text=best_string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    return f"""
        <div style='font-family: .AppleSystemUIFont, sans-serif;'>
          <div style='margin-bottom: 10px;'>
            <span style='color: #8E8E93; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;'>LOCAL MATCH</span>
            <span style='color: #30D158; font-size: 11px; margin-left: 8px; font-weight: 600; background-color: rgba(48, 209, 88, 0.15); padding: 2px 6px; border-radius: 4px;'>Score: {max_sim:.4f}</span>
          </div>

          <div style='color: #0A84FF; font-size: 16px; font-weight: 600; margin-bottom: 4px;'>
            📄 {file_name}
          </div>

          <div style='color: #8E8E93; font-size: 12px; margin-bottom: 14px;'>
            Source: <code style='background-color: rgba(255,255,255,0.06); padding: 2px 5px; border-radius: 4px;'>{best_path}</code>
          </div>

          <div style='background-color: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 14px; color: #E5E5E5; font-size: 13.5px; line-height: 1.5;'>{clean_text}</div>
        </div>
        """

import sys
if __name__=="__main__":
    app=QApplication(sys.argv)
    app.aboutToQuit.connect(watcher.stop)
    print("\n🛑 Stopping file watcher...")
    print("👋 System stopped safely.")
    widget=SearchWidget()
    widget.search_callback=search_engine_callback
    widget.show()
    sys.exit(app.exec())



