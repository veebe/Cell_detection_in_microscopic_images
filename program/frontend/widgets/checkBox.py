from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtCore import Qt

class CheckBoxWidget(QCheckBox):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 12px;
                font-weight: bold;
            }

            QCheckBox::indicator {
                border: 2px solid #7d7480;
                border-radius: 5px;
                width: 15px;
                height: 15px;
                background-color: #212020;
            }

            QCheckBox::indicator:checked {
                background-color: #803eb5;
                border-color: #803eb5;
            }

            QCheckBox::indicator:checked:hover {
                background-color: #9348cf;
                border-color: #664c7a;
            }

            QCheckBox::indicator:hover {
                border-color: #803eb5;
            }
        """)
