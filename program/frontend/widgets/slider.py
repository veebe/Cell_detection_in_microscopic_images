from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
import math

class SliderWidget(QWidget):
    def __init__(self, min_value=0, max_value=0, callback=None, inc_label=True, 
                 label_default='Value', percent=True, power_of_two=False):
        super().__init__()

        self.power_of_two = power_of_two
        self.inc_label = inc_label
        self.label_default = label_default
        self.percent = percent
        self.callback = callback
        
        if self.power_of_two:
            self.actual_min = 2 ** min_value
            self.actual_max = 2 ** max_value
            min_value = 0 
            max_value = int(math.log2(self.actual_max))

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

        self.label = QLabel(self._get_label_text())
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #ffffff; font-weight: bold;")

        if self.power_of_two:
            self.slider.valueChanged.connect(self._handle_power_callback)
        else:
            self.slider.valueChanged.connect(self._handle_regular_callback)

        self.slider.valueChanged.connect(self.update_label)

        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setMaximumHeight(60)

    def _get_label_text(self):
        value = self.get_value()
        
        if self.power_of_two:
            if self.inc_label:
                return f"{self.label_default}: {value}/{self.actual_max}"
            return f"{self.label_default}: {value}"
            
        if self.inc_label:
            return f"{self.label_default}: {self.slider.value() + 1}/{self.slider.maximum() + 1}"
        
        if self.percent:
            return f"{self.label_default}: {self.slider.value()}%"
            
        return f"{self.label_default}: {self.slider.value()}"

    def _handle_power_callback(self, exponent):
        if self.callback:
            self.callback(2 ** exponent)

    def _handle_regular_callback(self, value):
        if self.callback:
            self.callback(value)

    def update_label(self):
        self.label.setText(self._get_label_text())

    def get_value(self):
        if self.power_of_two:
            return 2 ** self.slider.value()
        return self.slider.value()

    def set_power_value(self, value):
        if self.power_of_two:
            exponent = int(math.log2(value))
            self.slider.setValue(exponent)

    def setValue(self, value):
        if self.power_of_two:
            self.set_power_value(value)
        else:
            self.slider.setValue(value)