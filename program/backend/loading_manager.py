import time
import multiprocessing
import threading
from frontend.widgets.loadScreen import loading_screen_process

class SharedState(object):
    def __init__(self):
        self.progress = multiprocessing.Value('i', 0)
        self.status = multiprocessing.Array('c', 100)
        self.frontend_loaded = multiprocessing.Value('b', False)
        self.backend_loaded = multiprocessing.Value('b', False)
        self.app_ready = multiprocessing.Value('b', False)

def progress_updater(shared_state):
    current_progress = 0
    
    shared_state.status.value = b"Loading UI components..."
    import random
    while not shared_state.frontend_loaded.value:
        if current_progress < 48:
            current_progress += random.randint(0, 3)
            shared_state.progress.value = current_progress
        time.sleep(0.2)
    
    shared_state.progress.value = 50
    shared_state.status.value = b"Loading application backend..."
    current_progress = 50
    
    while not shared_state.backend_loaded.value:
        if current_progress < 98:
            current_progress += random.randint(0, 3)
            shared_state.progress.value = current_progress
        time.sleep(0.2)
    
    shared_state.progress.value = 100
    shared_state.status.value = b"Starting application..."
    time.sleep(1)  

class LoadingManager:
    def __init__(self):
        self.shared_state = SharedState()
        self.loading_process = None
        
    def start_loading_screen(self):
        self.loading_process = multiprocessing.Process(
            target=loading_screen_process, 
            args=(self.shared_state,)
        )
        self.loading_process.start()
        
        updater_thread = threading.Thread(
            target=progress_updater, 
            args=(self.shared_state,)
        )
        updater_thread.daemon = True
        updater_thread.start()
        
        time.sleep(0.5)
        
    def notify_frontend_loaded(self):
        self.shared_state.frontend_loaded.value = True
        time.sleep(0.2)  
        
    def notify_backend_loaded(self):
        self.shared_state.backend_loaded.value = True
        time.sleep(0.2) 
        
    def finish_loading(self):
        self.shared_state.app_ready.value = True
        if self.loading_process:
            self.loading_process.join(timeout=1)
            
            if self.loading_process.is_alive():
                self.loading_process.terminate()