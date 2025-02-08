from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty

class ProgressBarWidget(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)  
        self.setTextVisible(True)  
        self.setMinimum(0)
        self.setMaximum(100)
        self._gradient_position = -20

        self.update_style()

        self.animation = QPropertyAnimation(self, b"gradient_position")
        self.animation.setStartValue(-20)
        self.animation.setEndValue(120)  
        self.animation.setDuration(1800)  
        self.animation.setLoopCount(-1) 
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.start()

    def update_style(self):
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #444;
                border-radius: 10px;
                background-color: #222;
                text-align: center;
                color: white;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                border-radius: 8px;
                background: qlineargradient(
                    spread:pad, 
                    x1:{self._gradient_position/100}, y1:0, 
                    x2:{(self._gradient_position+40)/100}, y2:0,
                    
                    stop:0 rgba(121, 186, 71, 255), 
                    stop:0.5 rgba(152, 222, 98, 255),
                    stop:1 rgba(121, 186, 71, 255)  
                );
            }}
        """)

    def get_gradient_position(self):
        return self._gradient_position

    def set_gradient_position(self, value):
        self._gradient_position = value
        self.update_style()

    gradient_position = pyqtProperty(int, get_gradient_position, set_gradient_position)