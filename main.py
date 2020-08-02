# This file contains the main class to run the model
import os
import math
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import time
import matplotlib.pyplot as plt

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, dataset, n_samples=10):
	# prepare fake examples
    X, _ = g_model.generate_fake_samples(n_samples)
    X = (X + 1) / 2.0
	# plot images
    f = plt.figure(figsize=(20,20))
    for i in range(10):
		# define subplot
        ax = plt.subplot(10, 1, 1 + i)
		# turn off axis
        # plt.axis('off')
		# plot raw pixel data
        ax.imshow(X[i, :, :])
    f.show()
	# save plot to file
	# filename1 = 'generated_plot_%04d.png' % (step+1)
	# plt.savefig(filename1)
	# plt.close()
	# evaluate the classifier model
    X, y = dataset
    _, acc = c_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    return acc

# train the generator and discriminator
def train(data_loader, model, n_iter = 100, epochs=10, n_batch=24, batch_size=8):
    # select supervised dataset
    iterator = data_loader.generate_supervised_samples(n_batch)
    # print(X_sup.shape, y_sup.shape)
    # model.c_load()
    print("### Start Training ###")
    finished = False
    history = []
    best_performance = 0
    tolerance = 0
	# manually enumerate epochs
    for i in range(n_iter):
        print("=======================================================")
        print("Training Procedure {0}".format(i+1))
        X_sup, y_sup = next(iterator)
        # calculate the size of half a batch of samples
        half_batch = int(X_sup.shape[0] / 2)
		# update supervised discriminator (c)
        print("supervised-real")
        [x_real, y_real], y_real2 = data_loader.generate_real_samples([X_sup, y_sup], half_batch)
        res = model.c_model.fit(x_real, y_real, batch_size=batch_size, epochs=epochs, validation_data=data_loader.test_data)
        performance = np.average(res.history['val_accuracy'])
        print("Average Performance: {0}".format(performance))
		# update unsupervised discriminator (d)
        print("unsupervised-real")
        model.d_model.fit(x_real, y_real2, batch_size=batch_size, epochs=epochs)
        print("unsupervised-fake")
        x_fake, y_fake = model.g_model.generate_fake_samples(half_batch)
        model.d_model.fit(x_fake, y_fake, batch_size=batch_size, epochs=epochs)

		# update generator (g)
        print("gan")
        x_gan, y_gan = model.g_model.generate_latent_points(X_sup.shape[0]), np.ones((X_sup.shape[0], 256, 256, 1))
        model.gan_model.fit(x_gan, y_gan, batch_size=batch_size, epochs=epochs)
        # evaluate the model performance every so often
        del X_sup
        del y_sup
        time.sleep(1)
        if i % 3 == 1:
            performance = summarize_performance(i, model.g_model, model.c_model, data_loader.test_data)
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
            print("Save new weight at iter {0}".format(i))
            model.c_save()
            model.gan_model.save_weights(model.save_path)
    print("Discriminator Training complete.")
    return finished, res.history['accuracy'], res.history['val_accuracy'], res.history['loss'], res.history['val_loss']



def train_supervised(model, data_loader, checkpoint_path = './', n_iter = 5000, n_batch = 24, batch_size=8, epochs=10):
    # model.load()

    # prepare training data loader
    iterator = data_loader.generate_supervised_samples(n_batch)
	# manually enumerate epochs
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
        res = model.c_model.fit(x_real, y_real, batch_size=batch_size, epochs=epochs, 
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
        time.sleep(1)
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
            model.c_save()
    print("Discriminator Training complete.")
    return finished, res.history['accuracy'], res.history['val_accuracy'], res.history['loss'], res.history['val_loss']