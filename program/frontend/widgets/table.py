from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt

class TableWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Epoch", "Loss", "Accuracy", "Val Loss", "Val Accuracy"])

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.verticalHeader().setVisible(False)

        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #222222; 
                color: white;              
                border: none;
                text-align: center;
            }

            QHeaderView::section {
                background-color: #333333; 
                color: white;
                font-weight: bold;
                text-align: center;
            }
        """)

        self.table.setMinimumSize(300,0)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        self.layout.addWidget(self.table)
        self.setLayout(self.layout)