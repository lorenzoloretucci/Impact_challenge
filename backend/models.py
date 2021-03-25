
import tensorflow as tf


def rnn_model(n_feat, step):
    
    obs_len = step

    '''
    Simple RNN model composed by one LSTM layer and two Dense.
    '''

    inputs = tf.keras.layers.Input((obs_len, n_feat))
    x = tf.keras.layers.LSTM(32)(inputs)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    out = tf.keras.layers.Dense(step, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=out)