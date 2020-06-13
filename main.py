# This file contains the main class to run the model
import os
import math
from tensorflow.keras.callbacks import LambdaCallback

def train(model, data_loader, checkpoint_path = './', n_iter = 1000, n_batch = 50):
    model.load()

    # prepare training data loader
    iterator = data_loader.generate_training_batches(n_batch)

	# manually enumerate epochs
    print("### Start Training ###")
    for i in range(n_iter):
        print("=======================================================")
        print("Training Procedure {0}".format(i+1))
		# get randomly selected 'real' samples
        x_real, y_real = next(iterator)
        # update discriminator on real samples
        model.train(x_real, y_real, batch_size=20, epochs=10, 
                    validation_data=data_loader.generate_testing_dataset(), callbacks = [LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_weights(-2)))])

        if i % 5 == 1 and not math.isnan(model.get_weights(-2)):
            print("Save new weight at iter {0}".format(i))
            model.save()
        # if i % 10 == 5 and not math.isnan(model.layers[-2].get_weights()[0][0][0][0][0]):
        #     print("Save new weight at iter {0}".format(i))
        #     model.save_weights(checkpoint_path2)
    print("Training complete.")