from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from models.basic import BasicModel

class Generator(BasicModel):
    def __init__(self, latent_dim = 100,
                 img_height = 128,
                 img_width = 128):
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width
        self.model = self._build_model()

    def _build_model(self):
        # build the layers
        x = Sequential()

        x.add(Dense(256 * 8 * 8, input_dim=latent_dim))
        x.add(LeakyReLU(alpha=0.2))
        x.add(Reshape((8, 8, 256)))

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
        x.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

        return x

    def summary(self): # provide model layer details
        return self.model.summary()

    def get_model(self):
        return self.model