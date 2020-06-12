# This file contains the main structure of the KAGAN model
from models.segmentation import SegmentationModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from loss.dk_loss import *

class Kegan(SegmentationModel):
    def __init__(self, img_height, img_width, class_num, optimizer, 
                 loss = 'mse', metrics = ['accuracy'], save_path='./kagan.h5', fcn_level = 32):
        super().__init__(save_path)
        self.img_height = img_height
        self.img_width = img_width
        self.fcn_level = fcn_level
        self.class_num = class_num

        self.model = self._build_model()
        self.model.compile(loss=loss, optimizer = optimizer, metrics=metrics)

    def train(self, x, y, batch_size=20, epochs=10, validation_data=None):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
    
    def predict(self, x):
        return np.argmax(self.model.predict(x), axis=3)

    def save(self):
        self.model.save_weights(self.save_path)
    
    def load(self, save_path=None):
        if save_path != None:
            self.model.load_weights(save_path)
            # save the new weights in its saving path
            self.save()
        else:
            self.model.load_weights(self.save_path)

    def _build_model(self):
        img_input = Input(shape=(self.img_height, self.img_width, 3))
        
        # use Res50 as base model
        base_model = ResNet50(weights=None, include_top=False, input_tensor=img_input)
        x = base_model.get_layer('conv5_block3_out').output

        # FCN-32 addition
        conv6 = Conv2D(name='conv6',
                        activation='relu',
                        filters=2048,
                        kernel_size=7,
                        padding='same',
                        data_format='channels_last'
                        )(x)
        conv7 = Conv2D(name='conv7', 
                        activation='relu',
                        filters=2048,
                        kernel_size=1,
                        padding='same',
                        data_format='channels_last'
                        )(conv6)
        # upsampling
        conv7_upsampling = Conv2DTranspose(name='conv7_upsampling',
                                            filters=self.class_num,
                                            kernel_size=4,
                                            strides=4,
                                            use_bias=False,
                                            data_format='channels_last'
                                            )(conv7)

        fcn32 = Conv2DTranspose(name='final_model32',
                                        filters=self.class_num,
                                        kernel_size=8,
                                        strides=8,
                                        use_bias=False,
                                        data_format='channels_last'
                                        )(conv7_upsampling)
                    
        if self.fcn_level == 32:  # return at fcn level 32
            fcn32 = (Activation('softmax'))(fcn32)
            return Model(img_input,fcn32)

        # FCN-16 addition
        # upsampling2
        y = base_model.get_layer('conv4_block3_out').output

        pool4_upsampling = Conv2D(name='pool4_upsampling',
                                    activation='relu',
                                    filters=self.class_num,
                                    kernel_size=1,
                                    padding='same',
                                    data_format='channels_last'
                                    )(y)
        pool4_upsampling2 = Conv2DTranspose(name='pool4_upsampling2',
                                            filters=self.class_num,
                                            kernel_size=2,
                                            strides=2,
                                            use_bias=False,
                                            data_format='channels_last'
                                            )(pool4_upsampling)

        if self.fcn_level == 16:  # return at fcn level 16
            fcn16 = Add(name='fcn_addition16')([pool4_upsampling2, conv7_upsampling])
            fcn16 = Conv2DTranspose(name='final_model16',
                                    filters=self.class_num,
                                    kernel_size=8,
                                    strides=8,
                                    use_bias=False,
                                    data_format='channels_last'
                                    )(fcn16)
            fcn16 = (Activation('softmax'))(fcn16)
            return Model(img_input,fcn16)

        # FCN-8 addition
        # upsampling3
        z = base_model.get_layer('conv3_block3_out').output
        pool3_upsampling = Conv2D(name='pool3_upsampling',
                                    activation='relu',
                                    filters=self.class_num,
                                    kernel_size=1,
                                    padding='same',
                                    data_format='channels_last'
                                    )(z)

        # Combination
        fcn8 = Add(name='fcn_addition8')([pool4_upsampling2,pool3_upsampling,conv7_upsampling])
        # upsampling4

        final = Conv2DTranspose(name='final_model8',
                                filters=self.class_num,
                                kernel_size=8,
                                strides=8,
                                use_bias=False,
                                data_format='channels_last'
                                )(fcn8)

        final = (Activation('softmax'))(final)

        return Model(img_input,final)  # return at fcn level 8

