from PyQt5.QtWidgets import QLabel

class LabelWidget(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
