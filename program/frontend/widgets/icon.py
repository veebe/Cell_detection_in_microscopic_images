from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtGui import QPainter, QPixmap


def create_colored_svg_icon(svg_path, color):

    renderer = QSvgRenderer(svg_path)
    pixmap = QPixmap(renderer.defaultSize())
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    renderer.render(painter)

    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)

    painter.end()

    return QIcon(pixmap)


class IconButtonWidget(QPushButton):
    def __init__(self, svg_path):
        super().__init__()

        self.svg_path = svg_path
        self.default_color = QColor(139, 130, 143) 

        self.icon = create_colored_svg_icon(self.svg_path, self.default_color)

        self.setIcon(self.icon)
        self.setIconSize(QSize(30, 30))
        self.setFixedSize(30, 30)
        self.setStyleSheet("QPushButton { border: None; background: transparent;}")

    def updateSVG(self,svg_path):
        self.svg_path = svg_path
        self.icon = create_colored_svg_icon(self.svg_path, self.default_color)
        self.setIcon(self.icon)

    def enterEvent(self, event):
        darker_color = self.default_color.darker(150) 
        self.setIcon(create_colored_svg_icon(self.svg_path, darker_color))
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setIcon(create_colored_svg_icon(self.svg_path, self.default_color))
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        darkest_color = self.default_color.darker(200) 
        self.setIcon(create_colored_svg_icon(self.svg_path, darkest_color))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setIcon(create_colored_svg_icon(self.svg_path, self.default_color))
        super().mouseReleaseEvent(event)
