import tensorflow as tf
from models import rnn_model


class ModelTraining:


    '''
    This class trains a simple RNN model on a given dataset.
    '''

    def train_loop(self, n_epochs):

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
    

    def training(self, 
                data=None, 
                step=30,
                epochs=20
                ):


        '''
        Basic data preparation and model setting. The data shape should be:
        (n_bins, n_obs, n_feat), where n_feat is the number of features. 
        The model predicts the next step states of the bins given the previous step observations.

        '''
        self.data = data
        self.step = step

        self.data = tf.cast(self.data, tf.int32)

        # use the last observations as validation data
        self.test_data = self.data[:, -step*2:, :]
        self.train_data = self.data[:, :-step*2, :]

       
        self.model = rnn_model(self.data.shape[2], self.step)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.acc = tf.keras.metrics.BinaryAccuracy()
        self.opt = tf.keras.optimizers.Adam()
        self.train_loop(epochs)
    
    def save_model_weights(self, PATH):
        '''
        Saves the weights of the model in a specified PATH.
        
        '''
        self.model.save_weights(PATH)
