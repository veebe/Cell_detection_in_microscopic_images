
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from frontend.widgets.processBar import ProgressBarWidget

def loading_screen_process(shared_state):
    app = QApplication([])
    
    screen = QWidget()
    screen.setWindowTitle("Loading...")
    screen.setFixedSize(300, 100)
    screen.setStyleSheet("background-color: #3C3737; color: white; font-size: 16px; font-weight: bold;")
    screen.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    
    layout = QVBoxLayout()
    
    progress_bar = ProgressBarWidget()
    progress_bar.setRange(0, 100)
    progress_bar.setValue(0)
    layout.addWidget(progress_bar)
    
    status_label = QLabel("Initializing application...")
    status_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(status_label)
    
    screen.setLayout(layout)
    screen.show()
    
    def update_ui():
        current_progress = shared_state.progress.value
        progress_bar.setValue(current_progress)
        
        status_text = shared_state.status.value.decode('utf-8').strip('\x00')
        if status_text:
            status_label.setText(status_text)
        
        if shared_state.app_ready.value:
            app.quit()
    
    timer = QTimer()
    timer.timeout.connect(update_ui)
    timer.start(50) 
    
    app.exec_()