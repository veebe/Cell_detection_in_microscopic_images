from PyQt5.QtWidgets import QPushButton, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont

class PurpleButton(QPushButton):
    def __init__(self,text="", parent=None):
        super().__init__(text, parent)
        
        self.setStyleSheet("""
            QPushButton {
                background-color: #803eb5;  
                color: white;               
                border-radius: 8px;    
                padding: 4px 10px;         
                font-size: 14px;          
                font-weight: bold;         
                border: 2px solid #660066; 
            }
            QPushButton:hover {
                background-color: #993399; 
            }
            QPushButton:pressed {
                background-color: #4B0082;
            }
            QPushButton:disabled {
                background-color: #745a75;
            }
        """)