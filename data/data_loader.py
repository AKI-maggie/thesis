import numpy as np
import os

# A prototype class which mentions basic data loading functions
class DataLoader():
    def __init__(self, img_path, label_path):
        self.img_p = img_path
        self.label_p = label_path
        
        # use a live loading for training samples later,
        # so the loader class would only save the ids of the 
        # training data
        if self._load() <= 0:
            print("No training data found.")
            os.quit()

    def _load(self):
        pass

# a data-loading class for SiftFlow data
class SiftFlowLoader(DataLoader):
    def __init__(self, img_path, label_path):
        super().__init__(img_path, label_path)
        
    def _load(self):
        for root, dirs, files in os.walk(self.img_p):
            for f in files:
                if f.endswith('.jpg'):
                    fid = f.split('.')[0]
                    self.imgs_id.append(fid)
        return len(self.img_ids)

    