import sys
from PyQt5.QtWidgets import QApplication
from ui_main import MainUI
from backend import CellDetectionController


def main():
    app = QApplication(sys.argv)

    ui = MainUI()

    controller = CellDetectionController(ui)

    ui.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
