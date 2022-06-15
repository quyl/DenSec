
import os
import numpy as np
import argparse
import keras
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.applications.densenet import DenseNet201
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model

from keras_utils2 import TransformerBlock, DenseNet, AddPositionEmbedding, MultiHeadSelfAttention
#from keras import DenseNet1
from tensorflow.keras import layers
from keras import regularizers


def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # Bottleneck layers
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)

    # Composite function
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)

    if drop_rate: x = Dropout(drop_rate)(x)

    return x


def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=3)

    return x

# transformer
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if is_max != 0:
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else:
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    return x

# densenet和transformer
def DenseNet_Transformer(nb_classes, input_shape):
    model=DenseNet(input_shape=(222,222), nb_classes=2, depth=10, growth_rate=25,
                                   dropout_rate=0.1, bottleneck=False, compression=0.5,origin_shape=input_shape).build_model()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.05, reduction="auto",
                                                      name="categorical_crossentropy")

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00000025,
                                                     amsgrad=False),
                  metrics=['accuracy']
                  )
    model.summary()
    return model

def PartNet(inputs, filters, kernel_size):
    drop_hid = 0.5

    x = Conv1D(filters, kernel_size,
                  strides=1, padding='same',
                  kernel_regularizer=regularizers.l2(0.01),
               )(inputs)
    x = Activation('relu')(x)
    x = Dropout(drop_hid)(x)


    embed_dim = filters
    num_heads = 4
    ff_dim = filters

    x_tmp = x

    embedding_layer = AddPositionEmbedding(1000, embed_dim)
    x = embedding_layer(x)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = Dropout(drop_hid)(x)

    x0 = GlobalAveragePooling1D()(x)

    x = x_tmp
    print(x)
    print('------------------------')
    x=Reshape((1000,16,1))(x)
    x=x[0,:,:,:]

    growth_rate = 12

    x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = BatchNormalization(axis=3)(x)
    x = GlobalAveragePooling2D()(x)
    print("x is shape")
    print(x.shape)
    x = Dropout(drop_hid)(x)
    x = Dense(16, activation='relu')(x)
    x = concatenate([x, x0], axis=-1)

    return x



