# This file contains the prototype for any image segementation model

class SegmentationModel():
    def __init__(self, save_path):
        self.save_path = save_path

    def train(self, x, y, batch_size=20, epochs=10, validation_data=None):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self, save_path=None):
        pass