import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,backend,activations
import numpy as np
from keras import regularizers
from keras.layers import *


def PartNet(inputs, filters):
    drop_hid = 0.7
    x=inputs
    embed_dim = filters
    num_heads = 3
    ff_dim = filters
    #x_tmp = x
    embedding_layer = AddPositionEmbedding(1000, embed_dim)
    x = embedding_layer(x)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = Dropout(drop_hid)(x)
    x0 = GlobalAveragePooling1D()(x)
    return x0
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__( **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
          
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim,kernel_regularizer=regularizers.l2(0.01))
        self.key_dense = layers.Dense(embed_dim,kernel_regularizer=regularizers.l2(0.01))
        self.value_dense = layers.Dense(embed_dim,kernel_regularizer=regularizers.l2(0.01))
        self.combine_heads = layers.Dense(embed_dim,kernel_regularizer=regularizers.l2(0.01))

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):        
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        #print(np.shape(attention))
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)

        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)

        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=16, num_heads=2, ff_dim=16, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu",kernel_regularizer=regularizers.l2(0.01)), layers.Dense(embed_dim,kernel_regularizer=regularizers.l2(0.01)),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        cfg = super().get_config()
        return cfg

class AddPositionEmbedding(layers.Layer):
    def __init__(self, maxlen=1000, embed_dim=16, **kwargs):
        super(AddPositionEmbedding, self).__init__(**kwargs)
        
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        
        return x + positions
        
    def get_config(self):
        cfg = super().get_config()
        return cfg    

def squash(x, axis=-1):
    s_squared_norm = backend.sum(backend.square(x), axis, keepdims=True)
    scale = backend.sqrt(s_squared_norm + backend.epsilon())
    return x / scale


class DenseNet:

    def __init__(self, input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None,
                 dropout_rate=None, bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40,origin_shape=None):
        """
        Arguments:
            input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
            dense_blocks : amount of dense blocks that will be created (default: 3)
            dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                           or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                           by the given depth (default: -1)
            growth_rate  : number of filters to add per dense block (default: 12)
            nb_classes   : number of classes
            dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                           In the paper the authors recommend a dropout of 0.2 (default: None)
            bottleneck   : (True / False) if true it will be added in convolution block (default: False)
            compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                           of 0.5 (default: 1.0 - will have no compression effect)
            weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
            depth        : number or layers (default: 40)
        """

        # Checks
        # if nb_classes == None:
        #     raise Exception('Please define number of classes (e.g. nb_classes=10). This is required for final softmax.')

        if compression <= 0.0 or compression > 1.0:
            raise Exception('Compression have to be a value between 0.0 and 1.0.')

        if type(dense_layers) is list:
            if len(dense_layers) != dense_blocks:
                raise AssertionError('Number of dense blocks have to be same length to specified layers')

        elif dense_layers == -1:
            dense_layers = int((depth - 4) // 3)
            if bottleneck:
                dense_layers = int(dense_layers // 2)
            dense_layers = [dense_layers for _ in range(dense_blocks)]
        else:
            dense_layers = [dense_layers for _ in range(dense_blocks)]
        self.origin_shape=origin_shape
        self.dense_blocks = dense_blocks
        self.dense_layers = dense_layers
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck
        self.compression = compression
        self.nb_classes = nb_classes

    def build_model(self):
        """
        Build the model

        Returns:
            Model : Keras model instance
        """
        drop_in = 0.2
        drop_hid = 0.7
        n_hid = 256
        inputs0 = Input(shape=self.origin_shape, name='input0')
        x = inputs0
        #x0,x_temp=PartNet(x,16,9)#transform  out
        # x = Conv1D(16, 9,
        #        strides=1, padding='same',
        #         kernel_regularizer=regularizers.l2(0.01),
        #         )(x)
        # x = Activation('relu')(x)
        # x = Dropout(drop_hid)(x)
        #x=GlobalAveragePooling1D()(x)
        #x = Bidirectional(LSTM(32))(x) 
        x = Reshape((1000, 20, 1))(x)
        ## x = x[0, :, :, :]
        print('Creating DenseNet')
        print('#############################################')
        print('Dense blocks: %s' % self.dense_blocks)
        print('Layers per dense block: %s' % self.dense_layers)
        print('#############################################')

        # img_input = layers.Input(shape=x_temp.shape, name='img_input')
        print("dafadfaf99999999999999999999")
        # print(img_input)
        nb_channels = self.growth_rate

        # Initial convolution layer
        x = layers.Convolution2D(2 * self.growth_rate, (3, 3), padding='same', strides=(1, 1),
                                 kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))(x)

        # Building dense blocks
        for block in range(self.dense_blocks - 1):
            # Add dense block
            x, nb_channels = self.dense_block(x, self.dense_layers[block], nb_channels, self.growth_rate,
                                              self.dropout_rate, self.bottleneck, self.weight_decay)

            # Add transition_block
            x = self.transition_layer(x, nb_channels, self.dropout_rate, self.compression, self.weight_decay)
            nb_channels = int(nb_channels * self.compression)

        # Add last dense block without transition but for that with global average pooling
        x, nb_channels = self.dense_block(x, self.dense_layers[-1], nb_channels,
                                          self.growth_rate, self.dropout_rate, self.weight_decay)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x=Reshape((31,3))(x)
        x=PartNet(x,3)
        x = Dense(256, activation='relu')(x)
        x = Dropout(drop_hid)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(drop_hid)(x)
        predictions = Dense(2, activation='softmax')(x)
        print('SDFGHJKL:')
        print(predictions.shape)
        # prediction = layers.Dense(16, activation='softmax')(x)
        # return x
        return tf.keras.Model(inputs=[inputs0], outputs=[predictions], name='densenet')

    def dense_block(self, x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False,
                    weight_decay=1e-4):
        """
        Creates a dense block and concatenates inputs
        """

        for i in range(nb_layers):
            cb = self.convolution_block(x, growth_rate, dropout_rate, bottleneck)
            nb_channels += growth_rate
            x = layers.concatenate([cb, x])
        return x, nb_channels

    def convolution_block(self, x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
        """
        Creates a convolution block consisting of BN-ReLU-Conv.
        Optional: bottleneck, dropout
        """

        # Bottleneck
        if bottleneck:
            bottleneckWidth = 4
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Convolution2D(nb_channels * bottleneckWidth, (1, 1),
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
            # Dropout
            if dropout_rate:
                x = layers.Dropout(dropout_rate)(x)

        # Standard (BN-ReLU-Conv)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(nb_channels, (3, 3), padding='same')(x)

        # Dropout
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        return x

    def transition_layer(self, x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
        """
        Creates a transition layer between dense blocks as transition, which do convolution and pooling.
        Works as downsampling.
        """

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(int(nb_channels * compression), (1, 1), padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

        # Adding dropout
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x

