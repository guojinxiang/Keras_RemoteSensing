from keras.models import Sequential, Graph
from keras.layers.core import Dense, Lambda, LambdaMerge, Activation, Layer
from keras.activations import relu
import theano.tensor as T
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def build_residual_block_dence(name, input_shape, input_name='x'):
    """
    Rough sketch of building blocks of layers for residual learning.
    See http://arxiv.org/abs/1512.03385 for motivation.
    """
    block = Graph()
    block.add_input(input_name, input_shape=input_shape)

    h1 = Dense(10, activation='relu')
    block.add_node(h1, name=name+'h1', input=input_name)

    h2 = Dense(10, activation='relu')
    block.add_node(h2, name=name+'h2', input=name+'h1')

    h3 = Dense(10, activation='linear')
    block.add_node(h3, name=name+'h3', input=name+'h2')

    block.add_output(name=name+'output', inputs=[name+'h1', name+'h3'], merge_mode='sum')

    return block

def build_residual_block_conv(name, input_shape, input_name='x'):
    """
    Rough sketch of building blocks of layers for residual learning.
    See http://arxiv.org/abs/1512.03385 for motivation.
    """
    block = Graph()
    block.add_input(input_name, input_shape=input_shape)

    h1 = Convolution2D(10, activation='relu')
    block.add_node(h1, name=name+'h1', input=input_name)

    h2 = Dense(10, activation='relu')
    block.add_node(h2, name=name+'h2', input=name+'h1')

    h3 = Dense(10, activation='linear')
    block.add_node(h3, name=name+'h3', input=name+'h2')

    block.add_output(name=name+'output', inputs=[name+'h1', name+'h3'], merge_mode='sum')

    return block