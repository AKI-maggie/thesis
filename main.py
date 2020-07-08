# This file contains the main class to run the model
import os
import math
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np

def train(model, data_loader, checkpoint_path = './', n_iter = 100, n_batch = 24, batch_size=8, epochs=10):
    # model.load()

    # prepare training data loader
    iterator = data_loader.generate_training_batches(n_batch)
    x_real, y_real = np.array([]), np.array([])
	# manually enumerate epochs
    model.d_load()
    print("### Start Training ###")
    for i in range(n_iter):
        print("=======================================================")
        print("Training Procedure {0}".format(i+1))
        x_real = None
        y_real = None
        del x_real
        del y_real
		# get randomly selected 'real' samples
        x_real, y_real = next(iterator)
        # update discriminator on real samples
        model.d_train(x_real, y_real, batch_size=batch_size, epochs=epochs, 
                    validation_data=data_loader.test_data)# callbacks = [LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_weights(-2)))])

        # if i % 5 == 1 and not math.isnan(model.get_weights(-2)):
            # print("Save new weight at iter {0}".format(i))
            # model.save()
        # if i % 10 == 5 and not math.isnan(model.layers[-2].get_weights()[0][0][0][0][0]):
        if i % 10 == 1:
            print("Save new weight at iter {0}".format(i))
            model.d_save()
    print("Discriminator Training complete.")

    # for i in range(n_iter):
    #     print("=======================================================")
    #     print("Training Procedure {0}".format(i+1))
	# 	# get randomly selected 'real' samples
    #     x_real, y_real = next(iterator)
    #     model.gan_train(x_real, y_real, batch_size=20, epochs=10, 
    #             validation_data=data_loader.generate_testing_dataset()) #callbacks = [LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_weights(-2)))])
