import os.path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from indexer import index_after_modification
import time
import threading
from indexer import delete_index
from PySide6.QtCore import QObject, Signal

class WatcherSignals(QObject):
    indexing_started=Signal()
    indexing_finished=Signal()

class Handler(FileSystemEventHandler):
    def __init__(self,model,threshold=5.0):
        self.model=model
        self.signals=WatcherSignals()
        self.active_timers={}
        self.lock=threading.Lock()
        self.time_threshold=threshold
        self.junk_signatures=(".swp",".swx",".swo",".tmp",">",".bak","~") #must correspond to accepted_file_types

    def handle_modification(self,path,timer):
        index=False
        with self.lock:
            if path in self.active_timers and self.active_timers[path] is timer:
                del self.active_timers[path]
                index=True
        if index:
            self.signals.indexing_started.emit()
            try:
                print(f"Indexing started for {path}...")
                index_after_modification(path,self.model,False)
                print(f"Indexing done for {path}...")
            finally:
                self.signals.indexing_finished.emit()
    def handle_creation(self,path,timer):
        index=False
        with self.lock:
            if path in self.active_timers and self.active_timers[path] is timer:
                del self.active_timers[path]
                index=True
        if index:
            self.signals.indexing_started.emit()
            try:
                print(f"Indexing started for {path}...")
                index_after_modification(path,self.model,True)
                print(f"Indexing done for {path}...")
            finally:
                self.signals.indexing_finished.emit()

    def handle_deletion(self,path):
        with self.lock:
            if path in self.active_timers:
                self.active_timers[path].cancel()
                del self.active_timers[path]
        print(f"Deleting index for {path}...")
        delete_index(path)
        print(f"Deletion done for {path}...")

    def on_modified(self,event):
        if not event.is_directory and not event.src_path.endswith(self.junk_signatures):
            print(f"Modification detected at {event.src_path}")
            with self.lock:
                if event.src_path in self.active_timers:
                    self.active_timers[event.src_path].cancel()
                new_timer=threading.Timer(self.time_threshold,self.handle_modification,args=[event.src_path,None])
                new_timer.args=[event.src_path,new_timer]
                self.active_timers[event.src_path]=new_timer
                new_timer.start()

    def on_created(self, event):
        if not event.is_directory and not event.src_path.endswith(self.junk_signatures):
            print(f"Creation detected at {event.src_path}")
            with self.lock:
                if event.src_path in self.active_timers:
                    self.active_timers[event.src_path].cancel()
                new_timer=threading.Timer(self.time_threshold,self.handle_creation,args=[event.src_path,None])
                new_timer.args=[event.src_path,new_timer]
                self.active_timers[event.src_path]=new_timer
                new_timer.start()


    def on_deleted(self,event):
        if not event.is_directory and not event.src_path.endswith(self.junk_signatures):
            print(f"Deletion detected at {event.src_path}")
            self.handle_deletion(event.src_path)

def start_file_watcher(folder_path,model):
    if not os.path.exists(folder_path):
        print(f"❌ Error: The directory '{folder_path}' does not exist.")
        return None, None
    print(f"🔍 Starting system. Monitoring changes in: {os.path.abspath(folder_path)}")
    handler=Handler(model)
    observer=Observer()
    observer.daemon=True
    observer.schedule(handler,path=folder_path,recursive=True)
    observer.start()
    print("✨ System is live and listening. Press Ctrl+C to exit.\n")
    return observer,handler


'''if __name__=="__main__":
    from main import model
    path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
    start_file_watcher(path,model)'''