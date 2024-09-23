from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.linear_model import RidgeClassifierCV

class BFClassifier(object):
    """
        The famous classier for segmenting audio files into background, 
        foreground, and background with foreground segments.
    """

    def __init__(self):
        super().__init__()
        # open the training data
        fa = open('datasets/features_BF90.csv','r')

        # header contains all feature names
        self.header = fa.readline().split(',')

        # load all data as np array
        bf_data = np.loadtxt(fa,delimiter=",")

        # carve the data
        train_y = bf_data[:,-1:]
        train_X = bf_data[:,0:-1]

        # verify correct training data length
        assert len(train_y) == len(train_X)

        # create model, scale the data using a pipeline 
        # computes the mean and standard deviation on the training set so as to be able to later re-apply the same transformation on the testing set
        self.pipe = make_pipeline(RidgeClassifierCV(normalize=True))

        # train the model
        self.pipe.fit(train_X, train_y.ravel()) 
        
        print('classifier initialized')
    
    def predict(self, features):
        return self.pipe.predict([features])

    def predict_prob(self, features):
        d = self.pipe.decision_function([features])
        probs = np.exp(d) / np.sum(np.exp(d))
        return probs[0]


    