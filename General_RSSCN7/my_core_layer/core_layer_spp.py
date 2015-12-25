# -*- coding:utf8 -*-
from __future__ import absolute_import
import theano
import numpy as np
import theano.tensor as T
import math

from keras.layers.core import activations,regularizers,constraints
from keras.layers.core import Layer
from keras.utils.theano_utils import shared_zeros

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class SppLayer(Layer):#耗时久
    def __init__(self,bins,feature_map_size=0):
        super(SppLayer,self).__init__()
        self.strides = []
        self.windows = []
        self.a = feature_map_size#feature_map size
        self.bins = bins
        self.num_bins = len(bins)

    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0],(self.bins(0)**2+self.bins(1)**2+self.bins(2)**2)*self.a)

    def get_output(self,train):
        self.input = self.get_input(train)
        for i in range(self.num_bins):
            self.strides.append(int(math.floor(self.a/self.bins[i])))
            self.windows.append(int(math.ceil(self.a/self.bins[i])))

        self.pooled_out = []
        for j in range(self.num_bins):
            self.pooled_out.append(downsample.max_pool_2d(input=self.input,
                                                              ds=(self.windows[j],self.windows[j]),
                                                              st=(self.strides[j],self.strides[j]),
                                                              ignore_border=False))

        for k in range(self.num_bins):
            self.pooled_out[k] = self.pooled_out[k].flatten(2)
            """
            print self.windows[k]
            print self.strides[k]
            print 'K: '+str(k)
            """
        # batch_size * image_size
        self.output = T.concatenate([self.pooled_out[0],self.pooled_out[1],self.pooled_out[2]],axis=1)
        return self.output