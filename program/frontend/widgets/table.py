from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from frontend.widgets.button import PurpleButton

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
        self.sort_column = 0  
        self.sort_order = Qt.AscendingOrder  

        self.table = QTableWidget()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        
        self.table.horizontalHeader().sortIndicatorChanged.connect(self.on_sort_changed)

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
        self.export_button = PurpleButton(text="Export table")
        self.export_button.clicked.connect(self.export_table)
        self.export_button.setToolTip("Export table to xlsx")  
        self.layout.addWidget(self.export_button)
        self.setLayout(self.layout)
    
    def on_sort_changed(self, column, order):
        self.sort_column = column
        self.sort_order = order

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
        # Remember the current sorting settings
        header = self.table.horizontalHeader()
        self.sort_column = header.sortIndicatorSection()
        self.sort_order = header.sortIndicatorOrder()
        
        was_sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        
        self.table.setRowCount(0)
        
        self._was_sorting_enabled = was_sorting_enabled
    
    def apply_sorting(self):
        if hasattr(self, '_was_sorting_enabled') and self._was_sorting_enabled:
            self.table.horizontalHeader().setSortIndicator(self.sort_column, self.sort_order)
            self.table.setSortingEnabled(True)
            delattr(self, '_was_sorting_enabled')

    def get_column_values(self, column):
        values = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, column)
            if item:
                values.append(item.text())
        return values
    
    def export_table(self):
        from backend.data.excel_utils import export_to_excel
        export_to_excel(self.table)
