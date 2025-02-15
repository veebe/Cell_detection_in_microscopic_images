from PyQt5.QtWidgets import QComboBox

class ComboBoxWidget(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;  
                color: white;  
                border: 1px solid #803eb5;
                border-radius: 5px;
                padding: 3px;
                font-size: 10px;
            }
            QComboBox::drop-down {
                border: 0px; 
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;  
                color: white;  
                border: 1px solid #ffffff;
                selection-background-color: #803eb5;
            }
        """)