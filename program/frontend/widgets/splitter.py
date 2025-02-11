from PyQt5.QtWidgets import QSplitter

class SplitterWidget(QSplitter):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSplitter::handle {
                background: #59535c;  
                height: 4px;  
            }

            QSplitter::handle:hover {
                background: #8a8a8a; 
            }

            QSplitter::handle:pressed {
                background: #803eb5;
            }
        """)