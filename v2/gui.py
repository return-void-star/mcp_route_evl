import sys
import os
import datetime

from PySide6.QtWidgets import (
    QApplication, QFrame, QVBoxLayout, QHBoxLayout, QLineEdit,
    QGraphicsDropShadowEffect, QWidget, QTextBrowser, QSystemTrayIcon,
    QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, Signal, QThread, QEvent, QTimer, QPropertyAnimation, QEasingCurve, QVariantAnimation
from PySide6.QtNetwork import QLocalServer
from PySide6.QtGui import QColor, QFont, QShortcut, QKeySequence, QPainter, QPen, QPixmap, QIcon
from PySide6.QtCore import QUrl

class SearchIconWidget(QWidget):
    """
    A custom vector widget that draws a clean, monochrome, translucent
    magnifying glass icon to match macOS Spotlight aesthetics.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Spotlight grey-white translucent icon color
        pen = QPen(QColor(255, 255, 255, 120), 2)
        painter.setPen(pen)
        
        # Draw magnifying glass circle (diameter 12, top-left at x=4, y=4)
        painter.drawEllipse(4, 4, 12, 12)
        
        # Draw handle starting from circle border at (13, 13) to (19, 19)
        painter.drawLine(13, 13, 19, 19)

class GlobalHotkeyThread(QThread):
    """
    Listens for Cmd+Shift+Space globally to toggle widget visibility.
    """
    toggle_signal = Signal()

    def run(self):
        try:
            from pynput import keyboard
        except ImportError:
            print("[Info] pynput is not installed. Global hotkey (Cmd+Shift+Space) will not work.")
            print("[Info] To enable global hotkey, run: pip install pynput")
            return

        def on_activate():
            self.toggle_signal.emit()

        # Listen for Cmd + Shift + Space on macOS (or Win + Shift + Space on Windows)
        try:
            with keyboard.GlobalHotKeys({'<cmd>+<shift>+<space>': on_activate}) as h:
                h.join()
        except Exception as e:
            print(f"[Warning] Failed to start global hotkey listener: {e}")

class SearchWidget(QFrame):
    def __init__(self):
        super().__init__()
        
        # 1. Initialize safety flags and state variables
        self._ignore_hide = False
        self.hotkey_thread = None

        # Set macOS activation policy to Accessory (overlay on top, no dock icon, no space switch)
        if sys.platform == "darwin":
            try:
                import AppKit
                AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
            except ImportError:
                pass
        
        # Setup local socket server for single-instance detection
        self.local_server = QLocalServer(self)
        self.local_server.removeServer("ai_asst_shortcut")
        self.local_server.listen("ai_asst_shortcut")
        self.local_server.newConnection.connect(self.handle_local_connection)
        
        # 2. Window configuration
        self.setWindowFlag(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 3. Layouts
        self.outer_layout = QVBoxLayout()
        self.outer_layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(self.outer_layout)
        
        self.container = QFrame()
        self.container.setObjectName("ContainerFrame")
        self.outer_layout.addWidget(self.container)
        
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(16, 12, 16, 12)
        self.container_layout.setSpacing(0)
        
        # Search bar top layout (horizontal: icon + input)
        self.search_bar_layout = QHBoxLayout()
        self.search_bar_layout.setContentsMargins(2, 0, 2, 0)
        self.search_bar_layout.setSpacing(12)
        
        self.search_icon = SearchIconWidget()
        self.search_bar_layout.addWidget(self.search_icon)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("")
        self.search_input.setFont(QFont(".AppleSystemUIFont", 20))
        self.search_input.setFixedHeight(40)
        self.search_input.setFrame(False)
        self.search_input.setAttribute(Qt.WA_MacShowFocusRect, False)
        
        # Style placeholder color via palette
        palette = self.search_input.palette()
        palette.setColor(palette.ColorRole.PlaceholderText, QColor(255, 255, 255, 80))
        self.search_input.setPalette(palette)
        
        self.search_bar_layout.addWidget(self.search_input)
        self.container_layout.addLayout(self.search_bar_layout)
        
        # Separator line (hidden by default)
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Plain)
        self.separator.setStyleSheet("background-color: rgba(255, 255, 255, 0.1); max-height: 1px; border: none; margin-top: 10px; margin-bottom: 5px;")
        self.separator.hide()
        self.container_layout.addWidget(self.separator)
        
        # Results viewer QTextBrowser (hidden by default)
        self.result_viewer = QTextBrowser()
        self.result_viewer.setFont(QFont(".AppleSystemUIFont", 14))
        self.result_viewer.setOpenLinks(False)
        self.result_viewer.anchorClicked.connect(self.on_anchor_clicked)
        self.result_viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set search path for local icons
        icons_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
        self.result_viewer.setSearchPaths([icons_path])
        
        self.result_viewer.hide()
        self.container_layout.addWidget(self.result_viewer)
        
        # Drop shadow effect (Spotlight-like deep, soft shadow)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(30)
        self.shadow.setColor(QColor(0, 0, 0, 160))
        self.shadow.setOffset(0, 8)
        self.container.setGraphicsEffect(self.shadow)
        
        # Stylesheet setup (generic styles)
        self.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: none;
                color: #FFFFFF;
                font-weight: 300;
                selection-background-color: #0A84FF;
            }
            QLineEdit:focus {
                border: none;
                background-color: transparent;
            }
            QTextBrowser {
                background-color: transparent;
                border: none;
                color: #E5E5E5;
                padding: 0px;
                font-family: sans-serif;
            }
            QTextBrowser td, QTextBrowser span, QTextBrowser div, QTextBrowser a {
                font-family: sans-serif;
            }
            QTextBrowser a {
                color: #FFFFFF;
                text-decoration: none;
                font-weight: normal;
            }
            QTextBrowser a:hover {
                color: #0A84FF;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.15);
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Keyboard shortcuts
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.activated.connect(self.hide)
        
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(QApplication.quit)
        self.quit_shortcut_mac = QShortcut(QKeySequence("Ctrl+W"), self)
        self.quit_shortcut_mac.activated.connect(QApplication.quit)

        # Alt/Option Shortcuts to avoid character input (like ∂, å, ƒ, ∑, ‰) in QLineEdit
        self.shortcut_chatgpt = QShortcut(QKeySequence("Alt+1"), self)
        self.shortcut_chatgpt.activated.connect(lambda: self.on_anchor_clicked(QUrl("action://escalate/chatgpt")))
        
        self.shortcut_gemini = QShortcut(QKeySequence("Alt+2"), self)
        self.shortcut_gemini.activated.connect(lambda: self.on_anchor_clicked(QUrl("action://escalate/gemini")))
        
        self.shortcut_claude = QShortcut(QKeySequence("Alt+3"), self)
        self.shortcut_claude.activated.connect(lambda: self.on_anchor_clicked(QUrl("action://escalate/claude")))
        
        self.shortcut_folder = QShortcut(QKeySequence("Alt+F"), self)
        self.shortcut_folder.activated.connect(lambda: self.on_anchor_clicked(QUrl("action://open_folder")))
        
        self.shortcut_weights = QShortcut(QKeySequence("Alt+W"), self)
        self.shortcut_weights.activated.connect(lambda: self.on_anchor_clicked(QUrl("action://view_weights")))
        
        self.shortcut_retrain = QShortcut(QKeySequence("Alt+R"), self)
        self.shortcut_retrain.activated.connect(lambda: self.on_anchor_clicked(QUrl("action://retrain")))
        
        self.shortcut_dash_d = QShortcut(QKeySequence("Alt+D"), self)
        self.shortcut_dash_d.activated.connect(self.toggle_actions_board)
        self.shortcut_dash_a = QShortcut(QKeySequence("Alt+A"), self)
        self.shortcut_dash_a.activated.connect(self.toggle_actions_board)
        
        # Window geometry setup (pixel-perfect height of 96px initial, will resize)
        self.setFixedSize(720, 96)
        self.center_on_screen()
        
        # Initialize and start global hotkey listener thread
        # self.hotkey_thread = GlobalHotkeyThread()
        # self.hotkey_thread.toggle_signal.connect(self.toggle_visibility)
        # self.hotkey_thread.start()
        
        # Install application-level event filter for Dock icon activations
        QApplication.instance().installEventFilter(self)
        self.installEventFilter(self)
        self.search_input.installEventFilter(self)
        self._ignore_hide = False

        # Initialize callbacks
        self.search_callback = None
        self.routing_correction_callback = None
        self.escalate_callback = None

        # Debounce timer for search typing (500ms delay)
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.execute_search)

        # Hook up search input text change event
        self.search_input.textChanged.connect(self.on_text_changed)

        # Typewriter placeholder animation with time-based greeting
        hour = datetime.datetime.now().hour
        if hour < 12:
            greeting = "Good morning..."
        elif hour < 17:
            greeting = "Good afternoon..."
        else:
            greeting = "Good evening..."

        self._tw_phrases = [
            greeting,
            "Search contacts...",
            "Search through documents...",
            "Ask anything...",
            "Find local knowledge...",
        ]
        self._tw_phrase_idx = 0
        self._tw_char_idx = 0
        self._tw_deleting = False
        self._tw_timer = QTimer(self)
        self._tw_timer.timeout.connect(self._tw_tick)

        # Smooth height animation (QVariantAnimation drives setFixedSize atomically)
        self._height_anim = QVariantAnimation(self)
        self._height_anim.setDuration(250)
        self._height_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._height_anim.valueChanged.connect(lambda v: self.setFixedSize(720, int(v)))



        # Drag tracking
        self._drag_pos = None

        # Initialize to empty state
        self.clear_results()
        self._tw_start()

    def set_tray_icon_color(self, color_hex):
        """Find the QSystemTrayIcon in the running application and change its dot color."""
        # Find all tray icons in the application
        tray_icons = QApplication.instance().findChildren(QSystemTrayIcon)
        if not tray_icons:
            return
            
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(color_hex))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(5, 5, 6, 6)
        painter.end()
        icon = QIcon(pixmap)
        
        for tray in tray_icons:
            tray.setIcon(icon)

    def update_glow(self, state):
        """Dynamically update border, drop shadow halo, and menu bar tray icon color."""
        if state == "empty":
            border_color = "rgba(255, 255, 255, 0.1)"
            shadow_color = QColor(0, 0, 0, 160)
            self.set_tray_icon_color("#30D158")  # Idle is green
        elif state == "local":
            border_color = "rgba(48, 209, 88, 0.2)"
            shadow_color = QColor(48, 209, 88, 50)
            self.set_tray_icon_color("#30D158")  # Local search is green
        elif state == "escalate":
            border_color = "rgba(10, 132, 255, 0.2)"
            shadow_color = QColor(10, 132, 255, 50)
            self.set_tray_icon_color("#0A84FF")  # Escalation turns blue!
        elif state == "low_confidence":
            border_color = "rgba(255, 69, 58, 0.2)"
            shadow_color = QColor(255, 69, 58, 50)
            self.set_tray_icon_color("#0A84FF")  # Low confidence/cloud turns blue!
        else:
            return

        self.container.setStyleSheet(f"""
            #ContainerFrame {{
                background-color: rgba(20, 20, 20, 0.94);
                border: 1px solid {border_color};
                border-radius: 16px;
            }}
        """)
        self.shadow.setColor(shadow_color)

    def show_empty_state(self):
        """Display a list of default Quick Actions when the query is empty."""
        system_font = "font-family: sans-serif;"
        
        html_content = f"""
        <div style='{system_font} padding: 2px;'>
          <div style='color: #8E8E93; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px; {system_font}'>⚡ Quick Actions</div>
          
          <table width='100%' border='0' cellspacing='0' cellpadding='4'>
            <tr>
              <td align='left' style='font-size: 13px; font-weight: 500; {system_font}'>
                <span style='color: #FFFFFF; vertical-align: middle; {system_font}'>📂 </span>
                <a href="action://open_folder" style="text-decoration: none; {system_font} vertical-align: middle;">Open Knowledge Folder</a>
              </td>
              <td align='right' style='font-size: 10px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                ⌥F
              </td>
            </tr>
            
            <tr><td height="8"></td></tr>
            
            <tr>
              <td align='left' style='font-size: 13px; font-weight: 500; {system_font}'>
                <span style='color: #FFFFFF; vertical-align: middle; {system_font}'>📊 </span>
                <a href="action://view_weights" style="text-decoration: none; {system_font} vertical-align: middle;">View Router Model Weights</a>
              </td>
              <td align='right' style='font-size: 10px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                ⌥W
              </td>
            </tr>
            
            <tr><td height="8"></td></tr>
            
            <tr>
              <td align='left' style='font-size: 13px; font-weight: 500; {system_font}'>
                <span style='color: #FFFFFF; vertical-align: middle; {system_font}'>🔄 </span>
                <a href="action://retrain" style="text-decoration: none; {system_font} vertical-align: middle;">Retrain Router Network</a>
              </td>
              <td align='right' style='font-size: 10px; font-weight: bold; font-family: monospace; color: #8E8E93;'>
                ⌥R
              </td>
            </tr>
          </table>
        </div>
        """
        self.result_viewer.setHtml(html_content)
        self.separator.show()
        self.result_viewer.show()
        
        # Calculate dynamic size
        self.result_viewer.document().setTextWidth(658)
        doc_height = int(self.result_viewer.document().size().height())
        target_height = 96 + doc_height + 15
        self._animate_to_height(target_height)
        
        self.update_glow("empty")

    def toggle_actions_board(self):
        """Toggle show/hide empty state dashboard when search is empty."""
        if self.result_viewer.isVisible() and self.search_input.text().strip() == "":
            self.clear_results()
        else:
            self.show_empty_state()

    def show_results(self, html_content):
        """Call this to display results and expand the window downward."""
        # Normalize font weights from main.py to keep links slim
        html_content = html_content.replace("font-weight: 500", "font-weight: normal")
        self.result_viewer.setHtml(html_content)
        self.separator.show()
        self.result_viewer.show()
        
        
        # Force a document layout pass by setting the text wrap width
        self.result_viewer.document().setTextWidth(658)
        doc_height = int(self.result_viewer.document().size().height())
        
        # Base height for input area + separator + margins is about 96px.
        # Add document height plus a small padding.
        target_height = 96 + doc_height + 15
        target_height = max(96, min(target_height, 650))
        
        self._animate_to_height(target_height)
        
        # Dynamic glow color logic based on parsing keywords in html_content
        if "LOCAL MATCH" in html_content:
            self.update_glow("local")
        elif "Low Confidence" in html_content or "⚠️" in html_content:
            self.update_glow("low_confidence")
        elif "Cloud Escalation" in html_content or "☁" in html_content:
            self.update_glow("escalate")
        else:
            self.update_glow("empty")

    def _animate_to_height(self, target_height):
        """Smoothly animate window height using atomic setFixedSize per frame."""
        current = self.height()
        if current == target_height:
            return
        self._height_anim.stop()
        self._height_anim.setStartValue(float(current))
        self._height_anim.setEndValue(float(target_height))
        self._height_anim.start()

    def clear_results(self):
        """Collapse widget back to compact search bar by default."""
        self.separator.hide()
        self.result_viewer.hide()
        self._animate_to_height(96)
        self.update_glow("empty")

    # ── Window Dragging ──────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    # ── Typewriter Placeholder Animation ──────────────────────────
    def _tw_start(self):
        """Start the typewriter cycling animation."""
        self._tw_phrase_idx = 0
        self._tw_char_idx = 0
        self._tw_deleting = False
        self._tw_timer.start(90)

    def _tw_stop(self):
        """Stop typewriter and clear placeholder."""
        self._tw_timer.stop()
        self.search_input.setPlaceholderText("")

    def _tw_tick(self):
        """One tick of the typewriter animation."""
        phrase = self._tw_phrases[self._tw_phrase_idx]

        if not self._tw_deleting:
            # Typing forward
            self._tw_char_idx += 1
            self.search_input.setPlaceholderText(phrase[:self._tw_char_idx])
            if self._tw_char_idx >= len(phrase):
                # Pause at full phrase, then start deleting
                self._tw_timer.stop()
                QTimer.singleShot(1800, self._tw_start_deleting)
        else:
            # Erasing backwards (faster)
            self._tw_char_idx -= 1
            self.search_input.setPlaceholderText(phrase[:self._tw_char_idx])
            if self._tw_char_idx <= 0:
                # Move to next phrase
                self._tw_deleting = False
                self._tw_phrase_idx = (self._tw_phrase_idx + 1) % len(self._tw_phrases)
                self._tw_timer.setInterval(90)

    def _tw_start_deleting(self):
        """Transition from typing to deleting."""
        self._tw_deleting = True
        self._tw_timer.setInterval(40)  # Erase faster than typing
        self._tw_timer.start()

    def on_text_changed(self, text):
        self.debounce_timer.stop()
        if not text.strip():
            self.clear_results()
            self._tw_start()  # Restart typewriter when input is cleared
            return
        self._tw_stop()  # Stop typewriter when user is typing
        # Wait 500ms after you stop typing to trigger search
        self.debounce_timer.start(500)

    def execute_search(self):
        query = self.search_input.text().strip()
        if not query or not self.search_callback:
            return
            
        # Execute the callback in main.py and get the HTML
        html_content = self.search_callback(query)
        if html_content:
            self.show_results(html_content)
        else:
            self.clear_results()

    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self.search_input.setFocus()

    def toggle_visibility(self):
        # If it's visible AND we are currently focused on it, hide it.
        # Otherwise (even if it's technically visible but on another macOS Desktop Space or behind another app), show it.
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        log_msg = f"{timestamp} - toggle_visibility called. isVisible={self.isVisible()}, isActiveWindow={self.isActiveWindow()}\\n"
        # We completely abandon the 'toggle' concept for the shortcut.
        # If the user presses the shortcut, they ALWAYS want to see the search bar.
        # Hiding is handled purely by clicking away or pressing Esc.
        self._ignore_hide = True
        self.show()
        self.raise_()
        self.activateWindow()
        self.search_input.setFocus()
        self.search_input.selectAll()
        
        if sys.platform == "darwin":
            try:
                import AppKit
                AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except ImportError:
                pass
                
        QTimer.singleShot(1000, self._clear_ignore_hide)


    def handle_local_connection(self):
        """Handle incoming local socket connections for single-instance window toggling."""
        socket = self.local_server.nextPendingConnection()
        if socket:
            socket.waitForReadyRead(100)
            msg = socket.readAll().data()
            if msg == b"toggle":
                self.toggle_visibility()
            socket.disconnectFromServer()

    def _clear_ignore_hide(self):
        self._ignore_hide = False

    def changeEvent(self, event):
        super().changeEvent(event)

    def eventFilter(self, obj, event):
        from PySide6.QtGui import QCursor
        
        if event.type() == QEvent.ApplicationActivate:
            self._ignore_hide = True
            self.show()
            self.raise_()
            self.activateWindow()
            self.search_input.setFocus()
            self.search_input.selectAll()
            QTimer.singleShot(1000, self._clear_ignore_hide)
        elif event.type() in (QEvent.ApplicationDeactivate, QEvent.WindowDeactivate):
            if not getattr(self, '_ignore_hide', False):
                # If mouse is touching the top menu bar, macOS might steal focus. Ignore it.
                if QCursor.pos().y() <= 25:
                    return super().eventFilter(obj, event)
                self.hide()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        # Let keyPressEvent pass through normally (Alt shortcuts are handled cleanly by QShortcut)
        super().keyPressEvent(event)

    def on_anchor_clicked(self, url):
        url_str = url.toString()
        if url_str == "action://correct_routing":
            query = self.search_input.text().strip()
            if query and self.routing_correction_callback:
                self.routing_correction_callback(query)
                self.execute_search()
        elif url_str == "action://go_back":
            self.show_empty_state()
        elif url_str.startswith("action://open_file/"):
            import subprocess
            file_path = url_str.replace("action://open_file/", "")
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", file_path])  # Reveal in Finder
            elif sys.platform == "win32":
                os.system(f'explorer /select,"{file_path}"')
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(file_path)])
        elif url_str.startswith("action://escalate/"):
            provider = url_str.split("/")[-1]
            query = self.search_input.text().strip()
            if query and self.escalate_callback:
                # Update GUI to show progress
                self.show_results(f"""
                    <div style='font-family: sans-serif; padding: 14px;'>
                      <div style='color: #0A84FF; font-weight: 600; font-size: 14px; font-family: sans-serif;'>🚀 Sending to Cloud</div>
                      <div style='color: #CCCCCC; font-size: 13px; margin-top: 6px; font-family: sans-serif;'>
                        Opening <b>{provider.capitalize()}</b> with your query and local memory pre-loaded...
                      </div>
                    </div>
                """)
                self.escalate_callback(query, provider)
        elif url_str == "action://open_folder":
            import os
            # Resolve path relative to project workspace root
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_dir, "data")
            if sys.platform == "darwin":
                os.system(f"open '{data_dir}'")
            elif sys.platform == "win32":
                os.system(f"explorer '{data_dir}'")
            else:
                os.system(f"xdg-open '{data_dir}'")
        elif url_str == "action://retrain":
            query = self.search_input.text().strip()
            if self.routing_correction_callback:
                # retrain network
                self.routing_correction_callback(query if query else "test")
                self.show_results("""
                    <div style='font-family: sans-serif; padding: 2px;'>
                      <div style='color: #30D158; font-size: 14px; font-weight: 600; margin-bottom: 8px; font-family: sans-serif;'>✅ Router Retrained Successfully!</div>
                      <div style='color: #8E8E93; font-size: 12px; line-height: 1.4; font-family: sans-serif;'>Model weights and biases have been updated in memory.</div>
                      <div style='margin-top: 12px; border-top: 1px solid #2D2D2D; padding-top: 8px; font-family: sans-serif;'>
                        <a href='action://view_weights' style='color: #0A84FF; font-size: 11px; text-decoration: none; font-weight: bold; font-family: sans-serif;'>📊 View New Weights</a>
                        <span style='color: #8E8E93; margin: 0px 8px;'>•</span>
                        <a href='action://go_back' style='color: #8E8E93; font-size: 11px; text-decoration: none; font-weight: bold; font-family: sans-serif;'>⚡ Go Back</a>
                      </div>
                    </div>
                """)
        elif url_str == "action://view_weights":
            import numpy as np
            import csv
            import os
            
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(project_dir, "v2", "router_config.npz")
            
            weights_info = "No weights found (please run retrain first)"
            bias_info = "0.000000"
            min_max_info = "N/A"
            samples_count = 0
            
            # Load queries count from CSV
            try:
                csv_path = os.path.join(project_dir, "router_queries.csv")
                with open(csv_path, "r") as f:
                    samples_count = sum(1 for _ in f) - 1
            except:
                pass
            weights_vector_html = ""
            
            if os.path.exists(weights_path):
                try:
                    data = np.load(weights_path)
                    w = data['w']
                    b = data['b']
                    bias_info = f"{float(b):.6f}"
                    min_max_info = f"{np.min(w):.5f} / {np.max(w):.5f}"
                    
                    # Format all 384 dimensions into 8 columns per line with line prefixes
                    lines = []
                    for i in range(0, len(w), 8):
                        chunk = w[i:i+8]
                        line_str = " ".join([f"{x:+.4f}" for x in chunk])
                        lines.append(f"{i:03d}:  {line_str}")
                    all_weights_text = "\n".join(lines)
                    
                    weights_vector_html = f"""
                    <div style='color: #8E8E93; font-size: 11px; margin-top: 10px; margin-bottom: 4px; font-family: sans-serif;'>Full Weights Vector ({w.shape[0]} dimensions):</div>
                    <pre style='background-color: #161618; border: 1px solid #2C2C2E; border-radius: 6px; padding: 10px; color: #30D158; font-family: monospace; font-size: 10px; line-height: 1.4; max-height: 110px;'>{all_weights_text}</pre>
                    """
                except Exception as e:
                    weights_vector_html = f"<div style='color: #FF453A; font-size: 12px; font-family: sans-serif;'>Error loading weights: {e}</div>"
            
            system_font = "font-family: sans-serif;"
            self.show_results(f"""
                <div style='{system_font} padding: 2px;'>
                  <div style='color: #FFFFFF; font-size: 14px; font-weight: 600; margin-bottom: 8px; {system_font}'>📊 Router Neural Network Parameters</div>
                  
                  <table width='100%' border='0' cellspacing='0' cellpadding='4' style='font-size: 12px; {system_font}'>
                    <tr>
                      <td style='color: #8E8E93; {system_font}'>Bias (b):</td>
                      <td style='color: #30D158; font-weight: 500; font-family: monospace;'>{bias_info}</td>
                    </tr>
                    <tr>
                      <td style='color: #8E8E93; {system_font}'>Min / Max Weight:</td>
                      <td style='color: #FFFFFF; font-weight: 500; font-family: monospace;'>{min_max_info}</td>
                    </tr>
                    <tr>
                      <td style='color: #8E8E93; {system_font}'>Trained Samples:</td>
                      <td style='color: #FFFFFF; font-weight: 500; {system_font}'>{samples_count} queries</td>
                    </tr>
                  </table>
                  
                  {weights_vector_html}
                  
                  <div style='border-top: 1px solid #2D2D2D; padding-top: 8px; margin-top: 12px; {system_font}'>
                    <a href='action://go_back' style='color: #8E8E93; font-size: 11px; text-decoration: none; font-weight: bold; {system_font}'>⚡ Go back</a>
                  </div>
                </div>
            """)
if __name__ == "__main__":
    print("Starting Terminal AI Assistant (Spotlight Widget)...")
    print("Press Esc to hide, and Ctrl+Q (or Ctrl+W) to quit.")
    app = QApplication(sys.argv)
    widget = SearchWidget()
    widget.show()
    sys.exit(app.exec())
