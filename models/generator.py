from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from models.basic import BasicModel
from numpy.random import randn
import numpy as np

class Generator(BasicModel):
    def __init__(self, class_num,
                 latent_dim = 100,
                 img_height = 128,
                 img_width = 128,
                 save_path = './g.h5'):
        super().__init__(save_path)
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width
        self.model = self._build_model()
        self.class_num = class_num

    def _build_model(self):
        # build the layers
        x = Sequential()

        x.add(Dense(128 * 16 * 16, input_dim=self.latent_dim))
        x.add(LeakyReLU(alpha=0.2))
        x.add(Reshape((16, 16, 128)))

        # deconvolutional layer
        x.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.2))

        x.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.2))

        x.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.2))

        x.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        x.add(BatchNormalization())
        x.add(LeakyReLU(alpha=0.2))

        # output layer
        x.add(Conv2D(3, (3, 3), activation='tanh', padding='same', name='g_output'))

        return x

    def summary(self): # provide model layer details
        return self.model.summary()

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples, seed = None):
        if seed:
            np.random.seed(seed)
        # generate points in the latent space
        z_input = randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, self.latent_dim)
        return z_input

    def generate_fake_samples(self, n_samples = 10, seed = None):
        # create fake image
        z_input = self.generate_latent_points(n_samples, seed)
        x = self.model.predict(z_input)

        # create fake label
        y = np.zeros([n_samples, self.img_height, self.img_width])

        return x, y
