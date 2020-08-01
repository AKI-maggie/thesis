# This file contains the main structure of the KAGAN model
from models.basic import BasicModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from numpy.random import randint

from models.generator import Generator
from loss.dk_loss import *
from loss.comb_loss import gan_activation
import os

from math import ceil, floor

class Kegan(BasicModel):
    def __init__(self, img_height, img_width, class_num, 
                 c_optimizer, 
                 d_optimizer,
                 g_optimizer,
                 c_loss = 'categorical_crossentropy', 
                 g_loss = 'binary_crossentropy',
                 d_loss = 'binary_crossentropy',
                 c_metrics = ['accuracy'], 
                 d_metrics = ['accuracy'],
                 g_metrics = ['accuracy'],
                 save_path='./kagan.h5', fcn_level = 32,
                 use_pyramid = True):

        # set basic configuration variables
        super().__init__(save_path)
        self.img_height = img_height
        self.img_width = img_width
        self.fcn_level = fcn_level
        self.class_num = class_num

        # build descriminator
        self.c_model, self.d_model = self._build_d_model(use_pyramid)
        self.c_model.compile(loss=c_loss, optimizer = c_optimizer, metrics=c_metrics)
        self.d_model.compile(loss=d_loss, optimizer = d_optimizer, metrics=d_metrics)

        # build generator
        self.g_model = self._build_g_model()

        # build gan structure
        self.gan_model = self._build_gan()
        self.gan_model.compile(loss=g_loss, optimizer=g_optimizer, metrics=g_metrics)
        
        # descriminator saving path
        if use_pyramid:
            self.c_save_path = os.path.join((os.path.split(save_path))[0], 's_d.h5')
        else:
            self.c_save_path = os.path.join((os.path.split(save_path))[0], 'd.h5')

    def c_load(self, save_path=None):
        if save_path != None:
            self.c_model.load_weights(save_path)
            # save the new weights in its saving path
            self.c_save()
        else:
            if os.path.exists(self.c_save_path):
                print("Pretrained weights found")
                self.c_model.load_weights(self.c_save_path)
            else:
                print("Pretrained weights not found")

    def c_save(self):
        self.c_model.save_weights(self.c_save_path)

    def gan_train(self, x, y, batch_size=20, epochs=10, validation_data=None, callbacks = []):
        bat_per_epo = int(x.shape[0] / batch_size)
        half_batch = int(batch_size / 2)

        for i in range(epochs):
            for j in range(bat_per_epo):
                # real train
                idx = randint(0, x.shape[0], half_batch)
                x_real = x[idx]
                y_real = y[idx]

                self.d_model.trainable = True
                d_loss1, _ = self.d_model.train_on_batch(x_real, y_real)
                # fake train
                x_fake, y_fake = self.g_model.predict(half_batch)
                d_loss2, _ = self.d_model.train_on_batch(x_fake, y_fake)
                # gan train
                self.d_model.trainable = False
                x_gan = self.g_model.generate_latent_points(half_batch)
                
                y_gan = np.zeros([half_batch, self.class_num + 1, self.img_height, self.img_width])
                y_gan[:, self.class_num, :, :] = 1
                g_loss = self.gan_model.train_on_batch(x_gan, y_gan.transpose(0, 2, 3, 1))

                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            if (i+1) % 10 == 0:
                self.summarize_performance(i, validation_data)

    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, epoch, dataset):
        # prepare real samples
        x_real, y_real = dataset
        # evaluate discriminator on real examples
        _, acc_real = self.d_model.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.g_model.predict(10)
        # evaluate discriminator on fake examples
        _, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        # save the generator model tile file
        # filename = 'generator_model_%03d.h5' % (epoch+1)
        # self.gan_model.save_weights(self.gan_save_path)
    
    def get_weights(self, layer_index):
        return self.d_model.layers[layer_index].get_weights()[0][0][0][0][0]

    def predict(self, x):
        return np.argmax(self.d_model.predict(x), axis=3)

    def _build_gan(self):
        # make weights in the discriminator not trainable
        self.d_model.trainable = False
        # connect image output from generator as input to discriminator
        gan_output = self.d_model(self.g_model.output)
        # define gan model as taking noise and outputting a classification
        return Model(self.g_model.input, gan_output)

    def _build_g_model(self):
        return Generator(self.class_num, img_height=self.img_height, img_width=self.img_width)

    def _build_d_model(self, use_pyramid):
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

        if use_pyramid:
            conv7 = self._build_ssp(conv7, conv7.shape[1])

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
            # supervised model
            fcn32_c = (Activation('softmax'))(fcn32)
            # unsupervised model
            fcn32_d = Lambda(gan_activation)(fcn32)

            return Model(img_input,fcn32_c), Model(img_input, fcn32_d)

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

            # supervised model
            fcn16_c = (Activation('softmax'))(fcn16)
            # unsupervised model
            fcn16_d = Lambda(gan_activation)(fcn16)

            return Model(img_input,fcn16_c), Model(img_input, fcn16_d)

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

        # supervised model
        final_c = (Activation('softmax'))(final)
        # unsupervised model
        final_d = Lambda(gan_activation)(final)

        return Model(img_input,final_c), Model(img_input, final_d)

    def _build_ssp(self, feature_map, last_shape):
        print(last_shape)
        # bin1
        ws1 = ceil(last_shape/1)  
        ss1 = floor(last_shape/1)
        # build 4 different bins from the feature map extracted by previous FCN
        bin1 = MaxPooling2D(name='spp1', pool_size=(ws1, ws1), strides=ss1)(feature_map)
        conv_bin1 = Conv2D(name='spp1_conv', activation='relu',kernel_size=1, padding='same',filters=1)(bin1)
        # upsample
        up_bin1 = Conv2DTranspose(name='spp1_up', filters=1, kernel_size=ws1, strides=ss1)(conv_bin1)

        # bin2
        ws2 = ceil(last_shape/2)
        ss2 = floor(last_shape/2)
        bin2 = MaxPooling2D(name='spp2', pool_size=(ws2, ws2), strides=ss2)(feature_map)
        conv_bin2 = Conv2D(name='spp2_conv', activation='relu',kernel_size=1, padding='same',filters=1)(bin2)
        # upsample
        up_bin2 = Conv2DTranspose(name='spp2_up', filters=1, kernel_size=ws2, strides=ss2)(conv_bin2)

        # bin3
        ws3 = ceil(last_shape/3)
        ss3 = floor(last_shape/3)
        bin3 = MaxPooling2D(name='spp3',pool_size=(ws3, ws3), strides=ss3)(feature_map)
        conv_bin3 = Conv2D(name='spp3_conv', activation='relu',kernel_size=1, padding='same',filters=1)(bin3)
        # upsample
        up_bin3 = Conv2DTranspose(name='spp3_up', filters=1, kernel_size=4, strides=ss3)(conv_bin3)

        # bin4
        ws4 = ceil(last_shape/6)+1
        ss4 = floor(last_shape/6)
        bin4 = MaxPooling2D(name='spp4',pool_size=(ws4, ws4), strides=ss4)(feature_map)
        conv_bin4 = Conv2D(name='spp4_conv', activation='relu',kernel_size=1, padding='same',filters=1)(bin4)
        # upsample
        up_bin4 = Conv2DTranspose(name='spp4_up', filters=1, kernel_size=ws4, strides=ss4)(conv_bin4)

        # concat
        result = Concatenate(axis=3)([feature_map, up_bin1, up_bin2, up_bin3, up_bin4])
        return result

