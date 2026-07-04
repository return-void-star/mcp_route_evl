# Local AI Knowledge Assistant & Cloud Router (v2)

A responsive, offline-first desktop search bar mimicking macOS Spotlight. It provides sub-second semantic document retrieval, automatic filesystem change indexing, and context-aware routing that decides whether to answer queries locally or escalate to cloud AI providers (ChatGPT, Gemini, Claude).

---

## 🏗️ System Architecture

The codebase separates presentation (GUI), orchestration (main execution loop), domain logic (vector/FTS retrieval and classification routing), and infrastructure (database and filesystem watching).

```
   +----------------------------------------------------------------+
   |                       USER INTERFACE                           |
   |              gui.py (SearchWidget : QFrame)                    |
   +-------------------------------+--------------------------------+
                                   | Signals / Callback
                                   v
   +----------------------------------------------------------------+
   |                      ORCHESTRATOR / DAEMON                     |
   |                            main.py                             |
   +-------+--------------------+-------------------+---------------+
           |                    |                   |
           v                    v                   v
+--------------------+ +------------------+ +-----------------------+
|    DOMAIN LAYER    | |   DOMAIN LAYER   | |  INFRASTRUCTURE LAYER |
|    retrieval.py    | |    router.py     | |     database.py       |
| (FTS5 + CosSim +   | | (Sigmoid weights | | (WAL connection pool |
|   RRF Merging)     | |  BCE gradients)  | |   SQLite Database)    |
+--------------------+ +------------------+ +-----------------------+
           ^                                                ^
           | Embedding Vector                               | Transaction
           v                                                v
+--------------------+                              +---------------+
| INFRASTRUCTURE     |                              | INFRASTRUCTURE|
| download_onnx.py   |                              |  indexer.py   |
| (ONNX Session CPU) |                              |  & watcher    |
+--------------------+                              +---------------+
```

### IPC & Process Flow (Single-Instance Singleton)
To ensure only one background daemon runs at a time, the launcher and daemon negotiate execution via a Unix Domain Socket:

```
[Keystroke: ⌥⌘Space] -> Executes 'python main.py' (Client Launcher Mode)
                             │
                             ▼
              Try connect to '/tmp/ai_asst_shortcut'
             /                                      \
    [Connection Succeeds]                     [Connection Fails]
            │                                         │
            ▼                                         ▼
   Send 'toggle' message                     Spawn Daemon Process:
   over Unix Domain Socket                   'python main.py --ai-bg-worker'
            │                                         │
            ▼                                         ▼
   Launcher Exits Instantly                  1. Initialize DB & Schema
                                             2. Load ONNX model
                                             3. Run background watcher
                                             4. Open QLocalServer
                                             5. Show Window
```

---

## 🛠️ Feature Deep-Dive & Mathematical Foundations

### 1. Local Semantic Embedding Generation (`download_onnx.py`)
To map text queries and document chunks into a unified vector space, we utilize the `all-MiniLM-L6-v2` transformer model exported to ONNX format.
*   **Tokenization**: Queries are split into subword tokens using a fast Rust-based Tokenizer matching the model vocabulary, truncating strings at a maximum length of 256 tokens.
*   **ONNX Inference**: The input vectors (`input_ids`, `attention_mask`, `token_type_ids`) are fed into the ONNX Runtime `InferenceSession` running on a CPU Execution Provider.
*   **Mean Pooling**: To convert raw token embeddings ($H$) into a single sentence vector, we perform mean pooling. We multiply the token embeddings by the expanded attention mask ($M$) and divide by the sum of the mask elements:
    $$\mathbf{v}_{\text{pooled}} = \frac{\sum_{i=1}^{L} H_i \cdot M_i}{\max(\sum_{i=1}^{L} M_i, 10^{-9})}$$
*   **Normalization**: The pooled vector $\mathbf{v}_{\text{pooled}}$ is normalized to unit length ($L2$ Norm) so that dot products yield cosine similarity scores:
    $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}_{\text{pooled}}}{\|\mathbf{v}_{\text{pooled}}\|_2}$$

---

### 2. Context-Aware Query Routing & Neuron Training (`train_router.py`)
A single-neuron neural network processes the 384-dimensional query vector to classify whether a query should run locally (Label `0`) or escalate to the cloud (Label `1`).

*   **Prediction Pipeline**:
    The input vector $\mathbf{x} \in \mathbb{R}^{384}$ is multiplied by the weight vector $\mathbf{w} \in \mathbb{R}^{384}$ plus bias $b \in \mathbb{R}$. The output is squashed through a Sigmoid activation function to yield the cloud probability $\hat{y}$:
    $$z = \mathbf{w} \cdot \mathbf{x} + b$$
    $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-\text{clip}(z, -400, 400)}}$$
    If $\hat{y} \ge 0.5$, the system triggers the cloud escalation action card.

*   **Training Mechanics**:
    The weights are trained using Gradient Descent on logs saved in `router_queries.csv`. The loss objective is Binary Cross-Entropy (BCE) regularized with an $L2$ penalty to prevent overfitting:
    $$\text{Loss} = -\frac{1}{M}\sum_{i=1}^{M} \left[ y_i \log(\hat{y}_i + \epsilon) + (1-y_i)\log(1-\hat{y}_i + \epsilon) \right] + \frac{\lambda}{2M}\|\mathbf{w}\|_2^2$$
    Where $\epsilon = 10^{-15}$ to prevent $\log(0)$ crashes, and $\lambda = 0.1$.
    
*   **Gradients**:
    During backward propagation, the gradients of the Loss with respect to weights ($\mathbf{w}$) and bias ($b$) are calculated as:
    $$\frac{\partial \text{Loss}}{\partial \mathbf{w}} = \frac{1}{M} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) + \frac{\lambda}{M} \mathbf{w}$$
    $$\frac{\partial \text{Loss}}{\partial b} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)$$
    The parameters are updated iteratively for 500 epochs with a learning rate $\alpha = 0.3$:
    $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial \text{Loss}}{\partial \mathbf{w}}$$
    $$b \leftarrow b - \alpha \frac{\partial \text{Loss}}{\partial b}$$

---

### 3. Two-Stage Bounded Hybrid Search (`retrieval.py`)
To combine semantic understanding and exact keyword searching, retrieval merges structural vector search with BM25 rankings:

1.  **Exact Phrase Filter**: The raw search input is cleaned into alphanumeric keywords. If keywords are present, the system runs an exact phrase query against the SQLite FTS5 contentless virtual table:
    ```sql
    SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH :phrase LIMIT 1
    ```
    If an exact phrase is matched, the system immediately returns it, bypassing further score calculations.
2.  **Cosine Similarity Matching**: The database calculates the dot product between the query vector and all chunk vectors stored in SQLite. The top 20 chunks are retained.
3.  **FTS5 Keyword Search**: An "OR" keyword search is executed on the FTS5 virtual table, returning the top 20 BM25 keyword matches.
4.  **Reciprocal Rank Fusion (RRF)**: The semantic rankings ($R_{\text{vector}}$) and keyword rankings ($R_{\text{keyword}}$) are combined:
    $$\text{RRF Score}(c) = \frac{1}{k + R_{\text{vector}}(c)} + \frac{1}{k + R_{\text{keyword}}(c)}$$
    Where constant $k = 60$. The chunk with the highest RRF Score wins.
5.  **Zero-Text Offset Slicing**: The system queries the database for the winning chunk's file path, start character index, and end character index. It dynamically reads and slices the document from disk:
    ```python
    with open(best_path, "r", encoding="utf-8") as f:
        text = f.read()
        chunk_text = text[char_start:char_end]
    ```

---

### 4. Background Indexing & Watchdog Watcher (`indexer.py` & `file_watcher.py`)
*   **Database Schema**:
    *   `docs`: Stores documents with path, parent folder, and file extension.
    *   `chunks`: Stores `doc_id`, `char_start`, `char_end`, `chunk_index`, and the 384-float vector BLOB.
    *   `chunks_fts`: Contentless FTS5 virtual table linking `rowid` to indexed search terms.
*   **Filesystem Monitor**:
    A watchdog observer monitors files in the background. On modification, addition, or deletion:
    1.  The watcher deletes the old records in `chunks` and `chunks_fts` for the modified file.
    2.  Reads the file, splits it into semantic chunks by tracking character indices, generates vectors, and inserts new records inside a safe transaction scope.

---

### 5. Spotlight Presentation Interface (`gui.py`)
*   **Frameless Translucency**: Uses window flags `Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool` and attribute `Qt.WA_TranslucentBackground`.
*   **Geometry**: Set at a fixed width of 720px. Height starts at 96px (only search bar visible) and dynamically resizes down to a maximum of 650px to display search results.
*   **Dynamic Glow Borders**: Adjusts border and drop-shadow halo colors based on the retrieval type:
    -   `Green Glow` (`#30D158`): High-confidence local knowledge match.
    -   `Blue Glow` (`#0A84FF`): Cloud escalation triggered or low-confidence match.
    -   `Grey Glow` (`rgba(255, 255, 255, 0.1)`): Empty input or system idle.
*   **Focus Loss Hiding**: Overrides `eventFilter` to intercept `QEvent.WindowDeactivate` and `QEvent.ApplicationDeactivate`. If you click away from the widget, it hides itself immediately to remain unobtrusive.

---

## 📋 System Requirements & Dependencies

### Operating System Support
*   **macOS**: Fully supported. Uses Apple AppKit's `NSApplicationActivationPolicyAccessory` to hide the Dock icon and keep the search window floating on top of all desktops.
*   **Windows**: Fully supported. System tray and window frames adapt to standard Windows scaling parameters.

### Python Environment Prerequisites
Ensure Python 3.12 or 3.13 is installed. The system requires the following libraries:

```text
PySide6>=6.5.0              # Qt GUI, Threading, and local sockets
onnxruntime>=1.15.0         # Running the ONNX embedding model
tokenizers>=0.13.0          # Subword text tokenization
numpy>=1.20.0               # Matrix dot products & weights arrays
watchdog>=3.0.0             # Background filesystem monitoring
pyobjc-framework-Cocoa>=9.0 # macOS native focus management (darwin only)
```

---

## 🏃 Run & Operation Guide

### 1. First Time Installation
First, install the library dependencies:
```bash
pip install PySide6 onnxruntime tokenizers numpy watchdog
# On macOS, also install the Objective-C bridging bindings:
pip install pyobjc-framework-Cocoa
```

Download the ONNX models from Hugging Face:
```bash
python download_onnx.py
```

### 2. Launching the Assistant (Daemon & Client)
Run the application. The first run will detect that no database exists, automatically crawl your local directory, generate embeddings, and spawn the background daemon:
```bash
python main.py
```

Once running, you will see a green status dot in your system menu bar tray.

### 3. Toggling the Search Bar (Setting Up Global Hotkeys)

To make the search bar trigger instantly from anywhere, you must configure a global keyboard shortcut on your OS to run the script:

#### macOS Configuration (via Shortcuts App)
1. Open the native **Shortcuts** app on your Mac.
2. Click the **+** (Create Shortcut) icon in the top-right toolbar.
3. Search for the **"Run Shell Script"** action in the action library and add it to your workflow.
4. Configure the shell action:
   - Set the Shell type (e.g., `/bin/zsh`).
   - Set the command to run Python against the absolute path to your script:
     ```bash
     /usr/local/bin/python3.13 /path/to/project/main.py
     ```
5. In the top-right sidebar, click the **Shortcut Details / Info** tab (sliders / ℹ️ icon).
6. Select **"Use as Quick Action"** and select **"Add Keyboard Shortcut"**.
7. Record your hotkey combination (e.g., `Option+Cmd+Space` or `Option+Space`).

#### Windows Configuration (via AutoHotkey)
1. Download and install [AutoHotkey](https://www.autohotkey.com/).
2. Create a file named `assistant.ahk` in your project folder.
3. Write the hotkey script (e.g., binding `Ctrl+Alt+Space` to execute `main.py`):
   ```autohotkey
   ^!Space::
   Run, python "C:\path\to\project\main.py", , Hide
   return
   ```
4. Double-click the `.ahk` script to run it, or place a shortcut to it in your Windows **Startup** folder (`shell:startup` in Windows Run) to load it automatically on boot.

Alternatively, you can toggle the search bar directly from your terminal:
```bash
python main.py
```
This client-execution script detects the running daemon, sends the `"toggle"` command over the Unix domain socket, and immediately exits, showing the search bar instantly.


### 4. Training and Rerouting
*   **Manual Retraining**: Press `Option+R` while the input is empty to force the system to retrain the neural routing network on your historical query CSV dataset.
*   **Forcing Route Modifications**: Click **"Reroute: Process locally"** or **"Reroute: Escalate to Cloud"** at the bottom of search results. This appends the correction to `router_queries.csv` and retrains the model in milliseconds.
