from PyQt5.QtWidgets import QMessageBox

class PopUpWidget:
    def __init__(self, msg_type, text):
        self.msg_type = msg_type.lower()  
        self.text = text

    def show(self):
        msg = QMessageBox()
        msg.setWindowTitle(self.msg_type.capitalize()) 
        msg.setText(self.text)

        if self.msg_type == "info":
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        elif self.msg_type == "warning":
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok) 
        elif self.msg_type == "error":
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)  
        elif self.msg_type == "question":
            msg.setIcon(QMessageBox.Question)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        else:
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        msg.setStyleSheet("""
            QMessageBox {
                background-color: #3C3737; 
                color: white; 
                font-size: 14px;
                font-weight: bold;
            }
            QMessageBox QLabel {
                color: white; 
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #803eb5; 
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px;
                border: 1px solid #660066;
            }
            QMessageBox QPushButton:hover {
                background-color: #993399;
            }
            QMessageBox QPushButton:pressed {
                background-color: #4B0082; 
            }
        """)

        return msg.exec_() 


