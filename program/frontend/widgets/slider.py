from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

class SliderWidget(QWidget):
    def __init__(self, min_value=0, max_value=0, callback=None, inc_label=True, label_default='Value', percent=True):
        super().__init__()

        self.inc_label = inc_label
        self.label_default = label_default
        self.percent = percent
        self.callback = callback
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_value, max_value)
        self.slider.setValue((min_value + max_value) // 2) 
        
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #888;
                background: gray;
                height: 6px;  
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #a32cc4;
                width: 8px;  
                height: 8px; 
                margin: -4px 0; 
                border-radius: 4px;
            }

            QSlider::handle:horizontal:hover {
                background: #803eb5;
                border: 2px solid #a32cc4; 
            }
        """)

        self.label = QLabel(f"{self.label_default}: {self.slider.value()}")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #ffffff; font-weight: bold;")

        self.slider.valueChanged.connect(self.update_label)
        if self.callback:
            self.slider.valueChanged.connect(self.callback)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.setMaximumHeight(60)

    def update_label(self):
        if self.inc_label:
            self.label.setText(f"Value: {self.slider.value() + 1}")
        else:
            if self.percent:
                self.label.setText(f"{self.label_default}: {self.slider.value()}%")
            else:
                self.label.setText(f"{self.label_default}: {self.slider.value()}")
