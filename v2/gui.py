import sys
import os

from PySide6.QtWidgets import (
    QApplication, QFrame, QVBoxLayout, QHBoxLayout, QLineEdit,
    QGraphicsDropShadowEffect, QWidget, QTextBrowser
)
from PySide6.QtCore import Qt, Signal, QThread, QEvent, QTimer
from PySide6.QtGui import QColor, QFont, QShortcut, QKeySequence, QPainter, QPen

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
    Listens for Alt+Space (Option+Space on macOS) globally to toggle widget visibility.
    """
    toggle_signal = Signal()

    def run(self):
        try:
            from pynput import keyboard
        except ImportError:
            print("[Info] pynput is not installed. Global hotkey (Alt+Space) will not work.")
            print("[Info] To enable global hotkey, run: pip install pynput")
            return

        def on_activate():
            self.toggle_signal.emit()

        # Listen for Alt + Space (maps to Option + Space on macOS)
        try:
            with keyboard.GlobalHotKeys({'<alt>+<space>': on_activate}) as h:
                h.join()
        except Exception as e:
            print(f"[Warning] Failed to start global hotkey listener: {e}")

class SearchWidget(QFrame):
    def __init__(self):
        super().__init__()
        
        # 1. Initialize safety flags and state variables first
        self._ignore_hide = False
        self.hotkey_thread = None
        
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
        self.search_input.setPlaceholderText("Search local knowledge...")
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
        self.result_viewer.hide()
        self.container_layout.addWidget(self.result_viewer)
        
        # Drop shadow effect (Spotlight-like deep, soft shadow)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 8)
        self.container.setGraphicsEffect(shadow)
        
        # Stylesheet setup
        self.setStyleSheet("""
            #ContainerFrame {
                background-color: rgba(25, 25, 25, 0.94);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 16px;
            }
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
        
        # Window geometry setup (pixel-perfect height of 96px)
        self.setFixedSize(720, 96)
        self.center_on_screen()
        
        # Initialize and start global hotkey listener thread
        self.hotkey_thread = GlobalHotkeyThread()
        self.hotkey_thread.toggle_signal.connect(self.toggle_visibility)
        self.hotkey_thread.start()
        
        # Install application-level event filter for Dock icon activations
        QApplication.instance().installEventFilter(self)
        self._ignore_hide = False

        # Initialize search callback
        self.search_callback = None

        # Debounce timer for search typing (500ms delay)
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.execute_search)

        # Hook up search input text change event
        self.search_input.textChanged.connect(self.on_text_changed)

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
        if self.isVisible() and self.isActiveWindow():
            self.hide()
        else:
            self._ignore_hide = True
            self.show()
            self.raise_()
            self.activateWindow()
            self.search_input.setFocus()
            self.search_input.selectAll()
            QTimer.singleShot(300, self._clear_ignore_hide)

    def _clear_ignore_hide(self):
        self._ignore_hide = False

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:
            if not self.isActiveWindow() and not getattr(self, '_ignore_hide', False):
                self.hide()
        super().changeEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.ApplicationActivate:
            self._ignore_hide = True
            self.show()
            self.raise_()
            self.activateWindow()
            self.search_input.setFocus()
            self.search_input.selectAll()
            QTimer.singleShot(300, self._clear_ignore_hide)
        return super().eventFilter(obj, event)

    def show_results(self, html_content):
        """Call this to display results and expand the window downward."""
        self.result_viewer.setHtml(html_content)
        self.separator.show()
        self.result_viewer.show()
        self.setFixedSize(720, 400)

    def clear_results(self):
        """Call this to hide the results pane and collapse the window back to compact."""
        self.separator.hide()
        self.result_viewer.hide()
        self.setFixedSize(720, 96)

    def on_text_changed(self, text):
        self.debounce_timer.stop()
        if not text.strip():
            self.clear_results()
            return
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

if __name__ == "__main__":
    print("Starting Terminal AI Assistant (Spotlight Widget)...")
    print("Press Esc to hide, and Ctrl+Q (or Ctrl+W) to quit.")
    app = QApplication(sys.argv)
    widget = SearchWidget()
    widget.show()
    sys.exit(app.exec())
