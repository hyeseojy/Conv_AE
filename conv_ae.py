'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-20
  email        : bao.salirong@gmail.com
  Task         : AlexNet based on Keras Model
'''

import tensorflow as tf
#==========================Conv_AE based on Keras Model=========================
class Conv_Encoder(tf.keras.Model):
    def __init__(self, units = 32, name = "bao_encoder"):
        super(Conv_Encoder, self).__init__(name = name)

        self.conv1 = tf.keras.layers.Conv2D(filters = units * 2,kernel_size=(3,3),strides=(2,2),padding = 'valid',activation = "relu")
        self.conv2 = tf.keras.layers.Conv2D(filters = units * 4,kernel_size=(3,3),strides=(2,2),padding = 'valid',activation = "relu")
        self.flatten= tf.keras.layers.Flatten()

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return x

class Conv_Decoder(tf.keras.Model):
    def __init__(self, input_shape, units = 32, name = "bao_decoder"):
        super(Conv_Decoder, self).__init__(name = name)

        c = input_shape[3]

        self.dense1 = tf.keras.layers.Dense(units = units * 100, activation = "relu")
        self.reshape   = tf.keras.layers.Reshape((5,5,units * 4), name = "de_main_out")
        self.conv1   = tf.keras.layers.Conv2DTranspose(filters = units * 2 , kernel_size=(3,3), strides=(2,2), padding='valid')
        self.conv2   = tf.keras.layers.Conv2DTranspose(filters = c , kernel_size=(4,4), strides=(2,2), padding='valid')

    def call(self, inputs):
        x = inputs
        x = self.dense1(x)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Conv_ae(tf.keras.Model):
    def __init__(self, latent = 100, units = 32, input_shape = None):
        super(Conv_ae, self).__init__()

        self.encoder = Conv_Encoder(units = units)
        self.la_dense= tf.keras.layers.Dense(units = latent, activation="relu")
        self.decoder = Conv_Decoder(input_shape = input_shape, units = units)

    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = self.la_dense(x)
        x = self.decoder(x)
        return x

#------------------------------------------------------------------------------
def Conv_AE(input_shape, latent, units):
    model = Conv_ae(latent = latent, units = units, input_shape = input_shape)
    model.build(input_shape = input_shape)
    return model
