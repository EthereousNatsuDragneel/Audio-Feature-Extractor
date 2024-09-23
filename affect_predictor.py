import numpy as np
from sklearn.ensemble import RandomForestRegressor

class AffectPredict:
    '''
        Emotion prediction models for valence and arousal.
    '''
    def __init__(self):
        # masks to select features
        self.AROUSAL_MASK = [11, 27, 29, 30, 33, 34, 48, 80, 89, 98, 110, 117, 118, 127, 128, 131, 133, 146, 166, 171, 203, 204, 219, 221, 236, 239, 261, 262, 264, 266, 267, 268, 343, 346, 347, 348, 349, 354, 355, 356, 364, 377, 379, 382, 383, 397, 437, 448, 450, 455, 463, 467, 468, 475, 478, 485, 487, 488, 491, 494, 497, 498, 508, 512, 518, 527, 529, 530, 531, 537, 539, 541, 544, 550, 557, 560, 573, 576, 580, 583, 584, 588, 596, 599, 602]
        self.VALENCE_MASK = [0, 12, 27, 29, 31, 32, 33, 35, 37, 39, 42, 45, 46, 48, 49, 53, 55, 75, 79, 86, 87, 89, 91, 93, 95, 104, 106, 118, 127, 131, 133, 134, 135, 138, 140, 148, 152, 155, 156, 157, 158, 159, 161, 166, 175, 180, 182, 183, 185, 186, 187, 188, 189, 195, 197, 200, 206, 207, 216, 218, 221, 227, 231, 242, 245, 248, 249, 254, 255, 256, 257, 264, 269, 270, 334, 335, 337, 339, 341, 353, 354, 357, 360, 364, 366, 367, 369, 370, 371, 373, 374, 376, 377, 378, 380, 381, 382, 384, 388, 389, 391, 393, 395, 398, 411, 412, 415, 416, 417, 418, 419, 423, 424, 425, 431, 432, 433, 435, 436, 437, 438, 443, 445, 449, 451, 458, 461, 462, 466, 468, 469, 470, 471, 473, 474, 475, 481, 482, 483, 485, 488, 489, 496, 497, 498, 499, 501, 505, 517, 518, 521, 523, 526, 528, 529, 530, 534, 535, 537, 539, 540, 542, 543, 544, 545, 549, 550, 553, 554, 555, 560, 563, 568, 573, 575, 577, 580, 583, 584, 585, 586, 599, 602]
        
        fa = open('datasets/arousal_data.csv','r')
        fv = open('datasets/valence_data.csv','r')

        self.arousal_header = fa.readline().split(',')
        self.valence_header = fv.readline().split(',')

        arousal_data = np.loadtxt(fa,delimiter=",")
        valence_data = np.loadtxt(fv,delimiter=",")

        self.arousal_y = arousal_data[:,-1:]
        self.arousal_X = arousal_data[:,0:-1]
        self.valence_y = valence_data[:,-1:]
        self.valence_X = valence_data[:,0:-1]

        # verify correct training data length
        assert len(self.arousal_X) == len(self.arousal_y)
        assert len(self.valence_X) == len(self.valence_y)

        # apply mask to get select features only
        self.arousal_X = [x[self.AROUSAL_MASK] for x in self.arousal_X]
        self.valence_X = [x[self.VALENCE_MASK] for x in self.valence_X]

        # create arousal model
        self.arousal_model = RandomForestRegressor(max_depth=20, min_samples_split=5, oob_score=True)
        self.arousal_model.fit(self.arousal_X, self.arousal_y.ravel())

        # create valence model
        self.valence_model = RandomForestRegressor(max_depth=30, min_samples_leaf=2, min_samples_split=5, oob_score=True)
        self.valence_model.fit(self.valence_X, self.valence_y.ravel())

    def predict_valence(self, Z):
        return self.valence_model.predict([Z]).item(0)

    def predict_arousal(self, Z):
        return self.arousal_model.predict([Z]).item(0)

    def model_stats(self):
        print('arousal r-squared score: %.2f' % self.arousal_model.oob_score_)
        print('valence r-squared score: %.2f' % self.valence_model.oob_score_)