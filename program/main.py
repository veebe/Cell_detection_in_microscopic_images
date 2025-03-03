import sys
import backend.suppress
from PyQt5.QtWidgets import QApplication
from backend.loading_manager import LoadingManager
import multiprocessing
import os

def main():
    loading_manager = LoadingManager()
    loading_manager.start_loading_screen()
    
    try:
        from frontend.ui_main import MainUI
        loading_manager.notify_frontend_loaded()
    except ImportError as e:
        print(f"Error importing frontend: {e}")
        loading_manager.finish_loading()
        return
    
    try:
        from backend.backend import CellDetectionController
        loading_manager.notify_backend_loaded()
    except ImportError as e:
        print(f"Error importing backend: {e}")
        loading_manager.finish_loading()
        return
    
    app = QApplication(sys.argv)
    ui = MainUI()
    controller = CellDetectionController(ui)
    
    loading_manager.finish_loading()
    
    ui.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if getattr(sys, 'frozen', False):   
        print("exe")
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
        
        bundle_dir = sys._MEIPASS
        os.chdir(bundle_dir)
        if bundle_dir not in sys.path:
            sys.path.insert(0, bundle_dir)
    else:    
        print("debug")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
    #with backend.suppress.SuppressStderr():
    main()