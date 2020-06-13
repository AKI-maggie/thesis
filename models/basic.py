# This file contains the prototype for any image segementation model
import os

class BasicModel():
    def __init__(self, save_path):
        self.save_path = save_path

    def train(self, x, y, batch_size=20, epochs=10, validation_data=None):
        pass

    def predict(self):
        pass

    def save(self):
        self.model.save_weights(self.save_path)
    
    def load(self, save_path=None):
        if save_path != None:
            self.model.load_weights(save_path)
            # save the new weights in its saving path
            self.save()
        else:
            if os.path.exists(self.save_path):
                print("Pretrained weights found")
                self.model.load_weights(self.save_path)
            else:
                print("Pretrained weights not found")