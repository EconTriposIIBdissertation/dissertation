from keras import backend as K
import keras
import numpy as np

class SemiSupervisedRegularizer(keras.regularizers.Regularizer):
    def __init__(self, supervised_weight_matrix, alpha):
        self.supervised_weight_matrix = K.variable(supervised_weight_matrix)
        self.alpha = K.cast_to_floatx(alpha)
        super(SemiSupervisedRegularizer, self).__init__()  
        
    def __call__(self,x):
        regularization = 0.
        regularization = (self.alpha) * K.sum(K.square(x - self.supervised_weight_matrix))
        return regularization

    def get_config(self):
        return {'supervised_weight_matrix':K.eval(self.supervised_weight_matrix),
               'alpha':float(self.alpha)}
