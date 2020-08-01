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
	# plot images
	for i in range(100):
		# define subplot
		plt.subplot(10, 10, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(X[i, :, :])
	# save plot to file
	# filename1 = 'generated_plot_%04d.png' % (step+1)
	# plt.savefig(filename1)
	# plt.close()
	# evaluate the classifier model
	X, y = dataset
	_, acc = c_model.evaluate(X, y, verbose=0)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))
	# save the generator model
	# filename2 = 'g_model_%04d.h5' % (step+1)
	# g_model.save(filename2)
	# save the classifier model
	# filename3 = 'c_model_%04d.h5' % (step+1)
	# c_model.save(filename3)
	# print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

# train the generator and discriminator
def train(data_loader, model, n_iter = 100, epochs=10, n_batch=24, batch_size=8):
	# select supervised dataset
	X_sup, y_sup = data_loader.generate_supervised_samples(n_batch)
	# print(X_sup.shape, y_sup.shape)
    model.c_load()

	# calculate the size of half a batch of samples
	half_batch = int(X_sup.shape[0] / 2)

	# manually enumerate epochs
	for i in range(n_iter):
		# update supervised discriminator (c)
		[x_real, y_real], y_real2 = data_loader.generate_real_samples([X_sup, y_sup], half_batch)
		c_loss, c_acc = model.c_model.fit(x_real, y_real, batch_size=batch_size, epochs=epochs, validation_data=data_loader.test_data)
		# update unsupervised discriminator (d)
		d_loss1 = model.d_model.fit(x_real, y_real, batch_size=batch_size, epochs=epochs)

		x_fake, y_fake = model.g_model.generate_fake_samples(half_batch)
		d_loss2 = model.d_model.fit(x_fake, y_fake, batch_size=batch_size, epochs=epochs)

		# update generator (g)
		x_gan, y_gan = model.g_model.generate_latent_points(X_sup.shape[0]), ones((X_sup.shape[0], 1))
		g_loss = model.gan_model.fit(x_gan, y_gan, batch_size=batch_size, epochs=epochs)
		# summarize loss on this batch
		print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
		# evaluate the model performance every so often
		if i % 3 == 1:
			summarize_performance(i, model.g_model, model.c_model, data_loader.test_data)


# def train(model, data_loader, checkpoint_path = './', n_iter = 5000, n_batch = 24, batch_size=8, epochs=10):
#     # model.load()

#     # prepare training data loader
#     iterator = data_loader.generate_training_batches(n_batch)
# 	# manually enumerate epochs
#     model.d_load()
#     print("### Start Training ###")
#     best_performance = 0
#     tolerance = 0
#     finished = False
#     history = []
#     for i in range(n_iter):
#         print("=======================================================")
#         print("Training Procedure {0}".format(i+1))
# 		# get randomly selected 'real' samples
#         x_real, y_real = next(iterator)
#         # update discriminator on real samples
#         res = model.d_train(x_real, y_real, batch_size=batch_size, epochs=epochs, 
#                     validation_data=data_loader.test_data)# callbacks = [LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_weights(-2)))])
#         performance = np.average(res.history['val_accuracy'])
#         loss = res.history['val_loss'][-1]
#         print("Average Performance: {0}".format(performance))
#         # if i % 5 == 1 and not math.isnan(model.get_weights(-2)):
#             # print("Save new weight at iter {0}".format(i))
#             # model.save()
#         # if i % 10 == 5 and not math.isnan(model.layers[-2].get_weights()[0][0][0][0][0]):
#         del x_real
#         del y_real
#         time.sleep(0.1)
#         if i % 3 == 1 and not math.isnan(loss):
#             print("Save new weight at iter {0}".format(i))
#             print("Best performance: {0}".format(best_performance))
#             print("Tolerance: {0}".format(tolerance))
#             if best_performance < performance:
#                 best_performance = performance
#                 tolerance = 0
#             elif best_performance - performance > 0.001:
#                 tolerance += 1
#             if tolerance > 3:
#                 print("Not progressing for too long time")
#                 finished = True
#                 break
#             model.d_save()
#     print("Discriminator Training complete.")
#     return finished, res.history['accuracy'], res.history['val_accuracy'], res.history['loss'], res.history['val_loss']

    # for i in range(n_iter):
    #     print("=======================================================")
    #     print("Training Procedure {0}".format(i+1))
	# 	# get randomly selected 'real' samples
    #     x_real, y_real = next(iterator)
    #     model.gan_train(x_real, y_real, batch_size=20, epochs=10, 
    #             validation_data=data_loader.generate_testing_dataset()) #callbacks = [LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_weights(-2)))])
