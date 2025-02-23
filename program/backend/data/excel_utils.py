import pandas as pd
from PyQt5.QtWidgets import QFileDialog
from frontend.widgets.popUpWidget import PopUpWidget

def export_to_excel(table):
    file_path, _ = QFileDialog.getSaveFileName(None, "Save table", "metrics_table.xlsx", "Excel Files (*.xlsx)")

    if not file_path:
        return 

    data = []
    for row in range(table.rowCount()):
        row_data = []
        for col in range(table.columnCount()):
            item = table.item(row, col)
            row_data.append(item.text() if item else "") 
        data.append(row_data)

    df = pd.DataFrame(data, columns=[table.horizontalHeaderItem(col).text() for col in range(table.columnCount())])

    df.to_excel(file_path, index=False)

    popup = PopUpWidget("info", f"Table exported successfully to: {file_path}")
    popup.show()
