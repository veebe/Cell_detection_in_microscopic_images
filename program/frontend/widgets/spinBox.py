from PyQt5.QtWidgets import QSpinBox

class SpinBoxWidget(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyle()

    def setStyle(self):
        self.setStyleSheet("""
            QSpinBox {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 2px solid #803eb5;
                border-radius: 5px;
                padding: 2px 5px;
                font-size: 12px;
                font-weight: bold;
                min-height: 20spx;
            }

            QSpinBox::up-button, QSpinBox::down-button {
                width: 15px;
                background-color: #23272A;
                border: none;
                border-radius: 3px;
            }

            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #803eb5;
            }

            QSpinBox::up-arrow, QSpinBox::down-arrow {
                width: 8px;
                height: 8px;
            }
        """)
