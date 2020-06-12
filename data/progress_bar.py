import time, sys
from IPython.display import clear_output

class ProgressBar():
    def __init__(self):
        self.progress = 0.0
        self.bar_length = 20
        
    def update_progress(self, new_progress):
        self.progress = new_progress

        if isinstance(self.progress, int):
            self.progress = float(self.progress)
        if not isinstance(self.progress, float):
            self.progress = 0
            
        if self.progress < 0:
            self.progress = 0.0
        if self.progress >= 1:
            self.progress = 1.0

        block = int(round(self.bar_length * self.progress))

        clear_output(wait = True)
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (self.bar_length - block), self.progress * 100)
        print(text)