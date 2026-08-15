

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):


    def __init__(self, units, **kwargs):

        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='b'
        )

        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='u'
        )

    def call(self, x):

        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)

        ait = tf.tensordot(uit, self.u, axes=1)


        a = tf.nn.softmax(ait, axis=-1)


        a = tf.expand_dims(a, axis=-1)


        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):

        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_cnn_lstm_lnn_attention(lookback, in_dim=3, out_dim=5, use_pooling=True):

    inp = layers.Input(shape=(lookback, in_dim), name="input_layer")

    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn_conv1"
    )(inp)


    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn_conv2"
    )(x)


    if use_pooling:
        x = layers.MaxPooling1D(pool_size=2, name="cnn_pool")(x)


    x = layers.BatchNormalization(name="cnn_bn")(x)


    x = layers.LSTM(
        units=64,
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        name="lstm1"
    )(x)

    x = layers.LSTM(
        units=32,
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        name="lstm2"
    )(x)


    x = layers.Dropout(rate=0.3, name="lstm_dropout")(x)


    x = layers.Dense(
        units=64,
        activation="tanh",
        kernel_regularizer=regularizers.l2(1e-3),
        name="lnn_dense1"
    )(x)


    x = layers.Dense(
        units=32,
        activation="tanh",
        kernel_regularizer=regularizers.l2(1e-3),
        name="lnn_dense2"
    )(x)


    x = AttentionLayer(units=32, name="attention")(x)




    x = layers.Dense(
        units=32,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-3),
        name="output_dense"
    )(x)


    out = layers.Dense(
        units=out_dim,
        activation="linear",
        name="output_layer"
    )(x)

   
    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_LNN_Attention_3to5")


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mae"]
    )

    return model
