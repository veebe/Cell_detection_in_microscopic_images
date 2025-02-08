import sys
import backend.suppress
from PyQt5.QtWidgets import QApplication
from frontend.ui_main import MainUI
from backend.backend import CellDetectionController

def main():
    app = QApplication(sys.argv)

    ui = MainUI()

    controller = CellDetectionController(ui)

    ui.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    #with backend.suppress.SuppressStderr():
    main()
