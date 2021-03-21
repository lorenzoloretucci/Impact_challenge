# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Prediction:
    '''
        This class generates a dataset of binary observations, trains a simple
        RNN model and generates prediction on a test set.
        To generate the datset use the method data_generation(), to train the 
        model use the method model_training() and after the training it is 
        possible to generate the predictions with the method get_predictions().

    '''


    def __init__(self, n_bins=1000, n_obs=1000, change_prob=False):

        self.n_bins = n_bins
        self.n_obs = n_obs
        self.change_prob = change_prob

    
    def data_generation(self, n_bins, n_obs, change_prob):

        '''
        Generate a n_bins series of length n_bos Bernoulli trials.
        For each bin the probability p of 1 changes over time, and 
        the probability that it changes depends on prob_p_change.

        '''
        data = np.empty(shape=(n_bins, n_obs))
        
        for j in range(n_bins):

            prob_p_change = np.random.rand() if change_prob else 0
            p = np.random.rand()

            for i in range(n_obs):
                r = np.random.rand()
                p = np.random.rand() if r < prob_p_change else p
                data[j, i] = np.random.binomial(n=1, p=p, size=1)

        return data

    
    def rnn_model(self, obs_len, batch_size):

        '''
        Simple RNN model composed by one LSTM layer and two Dense.

        '''
        inputs = tf.keras.layers.Input((obs_len,2), batch_size=batch_size)
        x = tf.keras.layers.LSTM(8,return_sequences=True,
                                            )(inputs)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        out = tf.keras.layers.Dense(2, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=out)
    
    def model_training(self):

        '''
        Basic data preparation and model setting.

        '''
        self.dataset = self.data_generation(self.n_bins, self.n_obs, self.change_prob)
        x_t = tf.cast(self.dataset, tf.int32)
        x_t = tf.one_hot(x_t, depth=2)
        y_t = x_t[:, 1:]
        x_t = x_t[:, :-1]

        self.BATCH_SIZE = 64 if self.n_bins > 64 else self.n_bins//2
        data = tf.data.Dataset.from_tensor_slices((x_t, y_t))
        train_data = data.skip(self.BATCH_SIZE)
        val_data = data.take(self.BATCH_SIZE)
        train_data = train_data.shuffle(200).batch(self.BATCH_SIZE, drop_remainder=True)
        val_data = val_data.shuffle(200).batch(self.BATCH_SIZE, drop_remainder=True)

        self.model = self.rnn_model(x_t.shape[1], self.BATCH_SIZE)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        opt = tf.keras.optimizers.Adam()
        self.model.compile(loss=loss_fn, optimizer=opt, metrics='acc')
        self.model.fit(train_data, validation_data=val_data, epochs=10)


    def get_predictions(self, n_test_bins=256, test_len=10):

        '''
        Generates a test set containing test_len obesrvations of
        a number n_test_bins of bins. Returns with observations of size:
        (n_test_bins, test_len).

        '''

        test_data = self.data_generation(n_test_bins, self.n_obs-1, self.change_prob)
        test_data = tf.one_hot(test_data, depth=2)
        x_te = tf.data.Dataset.from_tensor_slices(test_data)
        x_te = x_te.batch(self.BATCH_SIZE, drop_remainder=True)

        preds_list = []

        batch_pred = self.model.predict(x_te)
        pred = tf.random.categorical(batch_pred[:,-1,:], num_samples=1)
        preds_list.append(pred)
        batch_pred = tf.round(batch_pred)


        for _ in range(10-1):
            batch_pred = self.model(batch_pred, training=False)
            pred = tf.random.categorical(batch_pred[:,-1,:], num_samples=1)
            preds_list.append(pred)
            batch_pred = tf.round(batch_pred)

        predictions = np.stack(preds_list, axis=1)

        return tf.reshape(predictions, (predictions.shape[:-1]))


