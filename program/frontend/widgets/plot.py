from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)

        self.figure.patch.set_facecolor("#2b2b2b")  
        self.ax.set_facecolor("#444444") 
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

        self.ax.set_title("Predicted masks")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)