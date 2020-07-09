# This file contains the main class to run the model
import os
import math
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import time

def train(model, data_loader, checkpoint_path = './', n_iter = 5000, n_batch = 24, batch_size=8, epochs=10):
    # model.load()

    # prepare training data loader
    iterator = data_loader.generate_training_batches(n_batch)
	# manually enumerate epochs
    model.d_load()
    print("### Start Training ###")
    best_performance = 0
    tolerance = 0
    finished = False
    history = []
    for i in range(n_iter):
        print("=======================================================")
        print("Training Procedure {0}".format(i+1))
		# get randomly selected 'real' samples
        x_real, y_real = next(iterator)
        # update discriminator on real samples
        res = model.d_train(x_real, y_real, batch_size=batch_size, epochs=epochs, 
                    validation_data=data_loader.test_data)# callbacks = [LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_weights(-2)))])
        performance = np.average(res.history['val_accuracy'])
        loss = res.history['val_loss'][-1]
        print("Average Performance: {0}".format(performance))
        # if i % 5 == 1 and not math.isnan(model.get_weights(-2)):
            # print("Save new weight at iter {0}".format(i))
            # model.save()
        # if i % 10 == 5 and not math.isnan(model.layers[-2].get_weights()[0][0][0][0][0]):
        del x_real
        del y_real
        time.sleep(0.1)
        if i % 3 == 1 and not math.isnan(loss):
            print("Save new weight at iter {0}".format(i))
            print("Best performance: {0}".format(best_performance))
            print("Tolerance: {0}".format(tolerance))
            if best_performance < performance:
                best_performance = performance
                tolerance = 0
            elif best_performance - performance > 0.001:
                tolerance += 1
            if tolerance > 3:
                print("Not progressing for too long time")
                finished = True
                break
            model.d_save()
    print("Discriminator Training complete.")
    return finished, res.history['accuracy'], res.history['val_accuracy'], res.history['loss'], res.history['val_loss']

    # for i in range(n_iter):
    #     print("=======================================================")
    #     print("Training Procedure {0}".format(i+1))
	# 	# get randomly selected 'real' samples
    #     x_real, y_real = next(iterator)
    #     model.gan_train(x_real, y_real, batch_size=20, epochs=10, 
    #             validation_data=data_loader.generate_testing_dataset()) #callbacks = [LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_weights(-2)))])
