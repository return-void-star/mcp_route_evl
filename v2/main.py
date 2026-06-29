import os
from download_onnx import ONNXEmbedder

model=ONNXEmbedder()

from database import init_db,DB_PATH,get_conn

import numpy as np

if not os.path.isfile(DB_PATH):
    init_db()

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
from PySide6.QtWidgets import QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt

def create_status_icon(color_hex):
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color_hex))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(4, 4, 16, 16)
    painter.end()
    return QIcon(pixmap)


if __name__=="__main__":
    app=QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    idle_icon=create_status_icon("#30D158")
    indexing_icon=create_status_icon("#FF9500")

    from indexer import index_first_run
    is_first_run = True
    with get_conn() as temp_conn:
        cursor = temp_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM docs")
        if cursor.fetchone()[0] > 0:
            is_first_run = False
    start_icon=indexing_icon if is_first_run else idle_icon
    tray = QSystemTrayIcon(start_icon, parent=app)
    tray.setToolTip("Terminal AI Assistant: Indexing changes..." if is_first_run else "Terminal AI Assistant: Idle")
    tray.show()
    app.processEvents()
    if is_first_run:
        print("🚀 First-time startup detected. Indexing entire folder...")
        index_first_run(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), model)
        tray.setIcon(idle_icon)
        tray.setToolTip("Terminal AI Assistant: Idle")
        print("✨ First-time indexing complete.")

    from file_watcher import start_file_watcher
    watcher, handler = start_file_watcher(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), model)


    widget=SearchWidget()
    widget.search_callback=search_engine_callback
    tray.activated.connect(lambda reason: widget.toggle_visibility() if reason == QSystemTrayIcon.Trigger else None)
    menu = QMenu()
    show_action = menu.addAction("Show Search Widget")
    show_action.triggered.connect(widget.toggle_visibility)
    quit_action = menu.addAction("Exit App")

    def safe_quit():
        if watcher:
            print("\n🛑 Stopping file watcher...")
            watcher.stop()
            print("👋 System stopped safely.")
        app.quit()


    quit_action.triggered.connect(safe_quit)
    tray.setContextMenu(menu)
    tray.show()
    if handler:
        handler.signals.indexing_started.connect(lambda: (
            tray.setIcon(indexing_icon),
            tray.setToolTip("Terminal AI Assistant: Indexing changes...")
        ))
        handler.signals.indexing_finished.connect(lambda: (
            tray.setIcon(idle_icon),
            tray.setToolTip("Terminal AI Assistant: Idle")
        ))

    sys.exit(app.exec())




