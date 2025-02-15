from PyQt5.QtWidgets import QTabWidget

class TabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTabWidget::pane { 
                background-color: #212020; 
                border: 3px solid #803eb5;
                border-top-right-radius: 5px;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QTabBar::tab { 
                background: #591d8a; 
                padding: 8px;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 120px; 
                color: #bfbfbf;
            }
            QTabBar::tab:selected { 
                background: #803eb5;  
                color: white;
            }
            QTabBar::tab:hover { 
                background: #993399;  
                color: #ffffff;
            }
        """)
