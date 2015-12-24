__author__ = 'Administrator'

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

'''
Target: To investigate the performance of DBN on SAT-4/SAT-6.
'''

