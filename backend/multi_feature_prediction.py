
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
    
    def data_generation(self, n_bins, n_obs, change_prob, multiple_features=False):

        '''

        Generate a n_bins series of length n_obs Bernoulli trials.
        For each bin the probability p of 1 changes over time, and 
        the probability that it changes depends on prob_p_change.
        multiple_features is a boolean parameter that controls the 
        addition of other features, for example they can represent the
        expected rubbish for that bin (evaluated as the average observed
        rubbish at all the previous times) or other aggregate informations.

        '''

        if multiple_features:
            data = np.empty(shape=(n_bins, n_obs, 2))
        else:
            data = np.empty(shape=(n_bins, n_obs, 1))
        
        for j in range(n_bins):

            prob_p_change = np.random.rand() if change_prob else 0
            p = np.random.rand()

            
            for i in range(n_obs):
                r = np.random.rand()
                p = np.random.rand() if r < prob_p_change else p
                
                if multiple_features:
                    temp = []
                    temp.append(np.random.binomial(n=1, p=p, size=1)[0])
                    temp = temp + [1]
                    data[j, i] = temp
                else:
                    data[j, i] = np.random.binomial(n=1, p=p, size=1)

        return data

    
    def rnn_model(self, obs_len, multiple_features):
        
        n_feat = 2 if multiple_features else 1

        '''
        Simple RNN model composed by one LSTM layer and two Dense.
        '''

        inputs = tf.keras.layers.Input((obs_len, n_feat))
        x = tf.keras.layers.LSTM(32)(inputs)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        out = tf.keras.layers.Dense(self.step, activation='sigmoid')(x)
        return tf.keras.Model(inputs=inputs, outputs=out)


    def train_model(self, n_epochs):

        '''
        Functions that performs the training of the model.
        step is the length of the time series to consider for 
        the training. 

        '''
        data_len = self.train_data.shape[1]
        for epoch in range(n_epochs):
            print('epoch: ', epoch+1)
            for i in range(1, data_len//self.step-1):
                x = self.train_data[:, (i-1)*self.step:i*self.step, :]
                y = self.train_data[:, i*self.step:(i+1)*self.step,  0:1]
                self.train_step(x,y)
            print('acc: ', self.acc(self.test_data[:, -self.step:, 0], 
                          self.model(self.test_data[:, -2*self.step:-self.step, :])).numpy())


    @tf.function
    def train_step(self, x, y):
        '''
        Basic training step.
        '''
        with tf.GradientTape() as tape:

            pred = self.model(x)
            loss = self.loss_fn(y, pred)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss
    

    def model_training(self, 
                       data=None, 
                       n_bins=256, 
                       n_obs=1000, 
                       change_prob=False, 
                       multiple_features=True, 
                       step=30,
                       epochs=20
                       ):


        '''
        Basic data preparation and model setting. If data are available it is possible
        to use them for model training by passing a matrix of shape: (n_bins, n_obs, n_feat),
        where n_feat is the number of features. If the dataset is not available the class will 
        automatically generate it. 
        The model predicts the next step states of the bins given the previous step observations.

        '''

        self.step = step
        # to have the same data both in the training and test case we
        # keep this parameter the same for both cases.
        self.change_prob = change_prob

        # again needed for the test generation
        self.n_obs = n_obs

        if data is not None:
            self.data = data
        else:
            self.data = self.data_generation(n_bins, n_obs, change_prob, multiple_features)

        self.data = tf.cast(self.data, tf.int32)

        # use the last observations as validation data
        self.test_data = self.data[:, -step*2:, :]
        self.train_data = self.data[:, :-step*2, :]

       
        self.model = self.rnn_model(step, multiple_features)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.acc = tf.keras.metrics.BinaryAccuracy()
        self.opt = tf.keras.optimizers.Adam()
        self.train_model(epochs)

    def get_predictions(self, test_len=10):

        '''
        Generates the next step predictions given step observations
        of the test set.
        '''
        pred = self.model(self.test_data[:, -self.step:, :])

        return pred
