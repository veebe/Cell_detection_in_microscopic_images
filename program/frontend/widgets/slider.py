from PyQt5.QtWidgets import QSlider, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont

class CustomSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setTickPosition(QSlider.TicksBelow)
        self.setTickInterval(10)
        self.setSingleStep(1)
        
        self.setStyleSheet("QSlider::handle:horizontal { background: transparent; }")

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        spacing = (self.width() - 20) / (self.maximum() - self.minimum()) * 10
        for i in range(self.minimum(), self.maximum() + 1, 10):
            x = int((i - self.minimum()) / (self.maximum() - self.minimum()) * (self.width() - 20)) + 10
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(x - 5, self.height() - 5, str(i))

        value = self.value()
        handle_x = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), value, self.width()
        )
        handle_y = self.height() // 2 

        handle_radius = 20
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QColor(0, 0, 255))  
        painter.drawEllipse(handle_x - handle_radius // 2, handle_y - handle_radius // 2, handle_radius, handle_radius)

        painter.setPen(QColor(0, 0, 0)) 
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(handle_x - 10, handle_y - 10, 20, 20, Qt.AlignCenter, str(value))