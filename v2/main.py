import os
import webbrowser

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
neuron_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_config.npz")
if not os.path.exists(neuron_path):
    training_needed=1
else:
    training_needed=0
test_show=0

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
from PySide6.QtWidgets import QApplication
from gui import SearchWidget

def search_engine_callback(query):
    query_vec=model.encode(query)
    escalation_prob=predict_cloud_prob(query_vec,weights,bias)
    icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
    chatgpt_logo = f"file://{os.path.join(icons_dir, 'chatgpt.png')}"
    gemini_logo = f"file://{os.path.join(icons_dir, 'gemini.png')}"
    claude_logo = f"file://{os.path.join(icons_dir, 'claude.png')}"
    system_font="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;"
    if(escalation_prob>=0.5):
        return f"""
                        <div style='{system_font} padding: 2px;'>
                          <div style='color: #8E8E93; font-size: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 14px;'>☁️ Cloud Escalation Actions</div>

                          <table width='100%' border='0' cellspacing='0' cellpadding='4'>
                            <!-- ChatGPT Row -->
                            <tr>
                              <td align='left' style='font-size: 14px;'>
                                <img src='{chatgpt_logo}' width='18' height='18' align='middle' style='margin-right: 8px;'>
                                <a href="action://escalate/chatgpt" style="text-decoration: none; color: #FFFFFF; {system_font} vertical-align: middle;">Ask ChatGPT</a>
                              </td>
                              <td align='right' style='font-size: 11px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                                ⌥1
                              </td>
                            </tr>

                            <tr><td height="10"></td></tr>

                            <!-- Gemini Row -->
                            <tr>
                              <td align='left' style='font-size: 14px;'>
                                <img src='{gemini_logo}' width='18' height='18' align='middle' style='margin-right: 8px;'>
                                <a href="action://escalate/gemini" style="text-decoration: none; color: #FFFFFF; {system_font} vertical-align: middle;">Ask Gemini</a>
                              </td>
                              <td align='right' style='font-size: 11px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                                ⌥2
                              </td>
                            </tr>

                            <tr><td height="10"></td></tr>

                            <!-- Claude Row -->
                            <tr>
                              <td align='left' style='font-size: 14px;'>
                                <img src='{claude_logo}' width='18' height='18' align='middle' style='margin-right: 8px;'>
                                <a href="action://escalate/claude" style="text-decoration: none; color: #FFFFFF; {system_font} vertical-align: middle;">Ask Claude</a>
                              </td>
                              <td align='right' style='font-size: 11px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                                ⌥3
                              </td>
                            </tr>
                          </table>
                          <div style='border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 10px; margin-top: 14px;'>
                            <a href='action://correct_routing' style='color: #FFFFFF; font-size: 12px; text-decoration: none; {system_font} font-weight: 600;'>🔄 Reroute: Process locally</a>
                          </div>
                        </div>
                        """

    max_sim,best_string,best_path=search_locally(query_vec,query)
    local_threshold=0.25

    if(max_sim<local_threshold):
        return f"""
                        <div style='{system_font} padding: 2px;'>
                          <div style='color: #FF453A; font-size: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;'>⚠️ Low Confidence Local Match ({max_sim:.4f})</div>
                          <div style='color: #8E8E93; font-size: 13px; margin-bottom: 14px;'>No matching local knowledge. Escalate to:</div>

                          <table width='100%' border='0' cellspacing='0' cellpadding='4'>
                            <!-- ChatGPT Row -->
                            <tr>
                              <td align='left' style='font-size: 14px;'>
                                <img src='{chatgpt_logo}' width='18' height='18' align='middle' style='margin-right: 8px;'>
                                <a href="action://escalate/chatgpt" style="text-decoration: none; color: #FFFFFF; {system_font} vertical-align: middle;">Ask ChatGPT</a>
                              </td>
                              <td align='right' style='font-size: 11px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                                ⌥1
                              </td>
                            </tr>

                            <tr><td height="10"></td></tr>

                            <!-- Gemini Row -->
                            <tr>
                              <td align='left' style='font-size: 14px;'>
                                <img src='{gemini_logo}' width='18' height='18' align='middle' style='margin-right: 8px;'>
                                <a href="action://escalate/gemini" style="text-decoration: none; color: #FFFFFF; {system_font} vertical-align: middle;">Ask Gemini</a>
                              </td>
                              <td align='right' style='font-size: 11px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                                ⌥2
                              </td>
                            </tr>

                            <tr><td height="10"></td></tr>

                            <!-- Claude Row -->
                            <tr>
                              <td align='left' style='font-size: 14px;'>
                                <img src='{claude_logo}' width='18' height='18' align='middle' style='margin-right: 8px;'>
                                <a href="action://escalate/claude" style="text-decoration: none; color: #FFFFFF; {system_font} vertical-align: middle;">Ask Claude</a>
                              </td>
                              <td align='right' style='font-size: 11px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                                ⌥3
                              </td>
                            </tr>
                          </table>
                          <div style='border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 10px; margin-top: 14px;'>
                            <a href='action://correct_routing' style='color: #FFFFFF; font-size: 12px; text-decoration: none; {system_font} font-weight: 600;'>☁️ Reroute: Escalate to Cloud</a>
                          </div>
                        </div>
                        """

    file_name=os.path.basename(best_path) if best_path else "Unknown File"
    clean_text=best_string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    return f"""
            <div style='{system_font}'>
              <div style='margin-bottom: 10px;'>
                <span style='color: #8E8E93; font-size: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.8px;'>LOCAL MATCH</span>
                <span style='color: #30D158; font-size: 12px; margin-left: 8px; font-weight: 600; background-color: #172517; padding: 2px 6px; border-radius: 4px;'>Score: {max_sim:.4f}</span>
              </div>
              <div style='color: #FFFFFF; font-size: 16px; font-weight: 600; margin-bottom: 4px;'>
                📄 <a href="action://open_file/{best_path}" style="text-decoration: none; color: #FFFFFF;">{file_name}</a>
              </div>
              <div style='color: #8E8E93; font-size: 12px; margin-bottom: 14px;'>
                Source: <code style='background-color: #1E1E1E; padding: 2px 5px; border-radius: 4px;'>{best_path}</code>
              </div>
              <div style='background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 8px; padding: 14px; color: #E5E5E5; font-size: 14px; line-height: 1.5;'>{clean_text}</div>

              <table border='0' cellspacing='0' cellpadding='0' style='margin-top: 14px; border-top: 1px solid #2D2D2D; padding-top: 10px; margin-bottom: 8px; width: 100%;'>
                <tr>
                  <td align='left' style='font-size: 13px; font-weight: 600; color: #8E8E93; {system_font}'>
                    ☁️ Ask Cloud:
                    <img src='{chatgpt_logo}' width='16' height='16' align='middle' style='margin-left: 8px; margin-right: 4px;'>
                    <a href='action://escalate/chatgpt' style='color: #FFFFFF; text-decoration: none; {system_font} vertical-align: middle; margin-right: 12px;'>ChatGPT</a>

                    <img src='{gemini_logo}' width='16' height='16' align='middle' style='margin-right: 4px;'>
                    <a href='action://escalate/gemini' style='color: #FFFFFF; text-decoration: none; {system_font} vertical-align: middle; margin-right: 12px;'>Gemini</a>

                    <img src='{claude_logo}' width='16' height='16' align='middle' style='margin-right: 4px;'>
                    <a href='action://escalate/claude' style='color: #FFFFFF; text-decoration: none; {system_font} vertical-align: middle;'>Claude</a>
                  </td>
                </tr>
              </table>

              <div style='border-top: 1px solid #2D2D2D; padding-top: 8px;'>
                <a href='action://correct_routing' style='color: #8E8E93; font-size: 12px; text-decoration: none; {system_font} font-weight: bold;'>☁️ Reroute: Escalate to Cloud</a>
              </div>
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
    painter.drawEllipse(5, 5, 6, 6)
    painter.end()
    return QIcon(pixmap)

import csv
from train_router import sigmoid_func
def routing_correction(query):
    global weights,bias,n_data
    z=np.dot(model.encode(query),weights)+bias
    curr_prediction=sigmoid_func(z)
    curr_prediction=0 if curr_prediction<0.5 else 1
    row=[query,1 if curr_prediction==0 else 0]
    if os.path.exists(neuron_path):
        os.remove(neuron_path)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"router_queries.csv"),mode="a",newline="",encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(row)
    train_and_save_neuron(neuron_path,model,500,0.1,0.3)
    n_data=np.load(neuron_path)
    weights=n_data["w"]
    bias=n_data["b"]

import urllib.parse
def escalate_callback(query,provider):
    query_vec=model.encode(query)
    sim,best_string,best_path=search_locally(query_vec,query)
    if best_string and sim>0.1:
        clean_context=best_string.strip()
        prompt=f"Using this local context from my computer:\n---\n{clean_context}\n---\n\nAnswer this query: {query}"
    else:
        prompt=query
    encoded_prompt=urllib.parse.quote(prompt)
    urls={"chatgpt": f"https://chatgpt.com/?q={encoded_prompt}","gemini": f"https://gemini.google.com/app?q={encoded_prompt}","claude": f"https://claude.ai/new?q={encoded_prompt}"}
    url=urls.get(provider,"https://chatgpt.com")
    webbrowser.open(url)

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
    widget.routing_correction_callback=routing_correction
    widget.escalate_callback=escalate_callback
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




