from backend.models import rnn_model


N_STEP = 10 # number of time stamps used to make the prediction
N_FEAT = 2 # number of features in our toy dataset

class MakePrediction:

    '''
    This class loads the NN model, accesses the database and makes
    predictions. The class needs as argument the directory DIR_PATH 
    of the library. After that the prediction is done through the method
    .prediction(). The prediction is computed on the latest available
    observations stored in the DB and is computed for the next N_STEP
    timestamps.
    '''

    def __init__(self, DIR_PATH):
        self.w_path = DIR_PATH + '/backend/model_weights/weight (2)'
        self.obs_path = DIR_PATH + '/DATABASE/latest_time_obs.csv'
        self.feat_path = DIR_PATH + '/DATABASE/add_features.csv'
        self.load_model()
        self.data_processing()

    def draw_bernoulli(self, p):

        '''
        In our model we get class probabilities predictions, 
        from which we extract random variables, i.e. if a 
        prediction is (0.2, 0.8) it means that the prediction
        is 1 with 80% confidence, and we then extract a Ber(0.8)
        random variable.

        '''
        return np.random.binomial(n=1, p=p, size=1)

    def data_processing(self):
        '''
        Loads the DB and does basic preprocessing to feed the
        data in the NN model.
        '''
        obs = pd.read_csv(self.obs_path)
        feat = pd.read_csv(self.feat_path)

        data = pd.merge(obs, feat, on='id')

        obs = data.iloc[:, 1:11].values
        feat = data.iloc[:,11:].values

        obs = obs.reshape(-1, 10, 1)
        feat = feat.reshape(-1, 10, 1 )

        self.data = np.concatenate((obs, feat), axis=2)

    def prediction(self):
        ''' 
        Computes the predictions using the loaded model.

        '''
        preds = self.model(self.data, training=False)
        vec_draw_bernoulli = np.vectorize(self.draw_bernoulli)
        preds = vec_draw_bernoulli(preds)
        return preds
    
    def load_model(self):
        '''
        Loads the model with weights that are stored in our DB.
        '''
        self.model = rnn_model(N_FEAT, N_STEP)
        self.model.load_weights(self.w_path)

