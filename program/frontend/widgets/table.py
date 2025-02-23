from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt

class NumericTableWidgetItem(QTableWidgetItem):
    def __init__(self, text):
        super().__init__(text)
        try:
            self.value = float(text)
        except ValueError:
            self.value = float('-inf') 

    def __lt__(self, other):
        try:
            return self.value < other.value
        except AttributeError:
            return super().__lt__(other)

class TableWidget(QWidget):
    def __init__(self, parent=None, columns=[], min_width=0):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.numeric_columns = set()  

        self.table = QTableWidget()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)

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

        if min_width == 0:
            self.table.setMinimumSize(250, 0)
        else:
            self.table.setMinimumSize(min_width, 0)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        self.layout.addWidget(self.table)
        self.setLayout(self.layout)

    def set_item(self, row, col, value):
        try:
            float(value)
            item = NumericTableWidgetItem(str(value))
            self.numeric_columns.add(col)
        except (ValueError, TypeError):
            item = QTableWidgetItem(str(value))
        
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, col, item)

    def add_row(self, row_data):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, value in enumerate(row_data):
            self.set_item(row, col, value)

    def clear_table(self):
        self.table.setRowCount(0)
        self.numeric_columns.clear()

    def get_column_values(self, column):
        values = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, column)
            if item:
                values.append(item.text())
        return values