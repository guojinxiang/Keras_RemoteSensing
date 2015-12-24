# -*- coding:utf8 -*-
from keras import initializations
from keras.layers.core import Layer, MaskedLayer
from keras.utils.theano_utils import shared_zeros, shared_ones, sharedX
import theano.tensor as T
import numpy as np

class relu(MaskedLayer):
    """
    no use ,权值不更新
    """
    def __init__(self, **kwargs):
        super(relu, self).__init__(**kwargs)
    def get_output(self, train):
        X = self.get_input(train)
        neg = (X - abs(X))*0
        pos = ((1/2)*(X + abs(X)))
        return (pos + neg)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                      "alpha": self.alpha}
        base_config = super(relu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Lrelu(MaskedLayer):
    """
    no use , 收敛太慢
    """
    def __init__(self, alpha=0.3, **kwargs):
        super(Lrelu, self).__init__(**kwargs)
        self.alpha = alpha
    def get_output(self, train):
        X = self.get_input(train)
        neg = self.alpha * ((X - abs(X))/2)
        pos = ((1/2)*(X + abs(X)))
        return (pos + neg)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                      "alpha": self.alpha}
        base_config = super(Lrelu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Prelu(MaskedLayer):
    def __init__(self, init='zero', weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(Prelu, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape[1:]
        self.alpha = self.init(input_shape)
        self.params = [self.alpha]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output(self, train):
        X = self.get_input(train)
        pos = ((1/2)*(X + abs(X)))
        neg = self.alpha * ((X - abs(X))/2)
        return pos + neg

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "init": self.init.__name__}
        base_config = super(Prelu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))