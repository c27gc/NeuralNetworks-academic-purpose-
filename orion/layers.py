import theano.tensor as tensor
import theano
import numpy as np

class Input():
    def __init__(self, number_of_inputs):
        self.number_of_inputs = number_of_inputs
        
class Dense():
    number_of_layers = 0
    def __init__(self, units, activation = 'linear', use_bias = True, kernel_initializer='uniform'):
        Dense.number_of_layers += 1 
        self.number_of_layers = Dense.number_of_layers

        if activation == 'linear':
            self.activation = lambda x: x
        elif activation == 'sigmoid':
            self.activation = theano.tensor.nnet.nnet.sigmoid
        elif activation == 'softplus':
            self.activation = theano.tensor.nnet.nnet.softplus
        elif activation == 'softmax':
            self.activation = theano.tensor.nnet.nnet.softmax
        elif activation == 'relu':
            self.activation = theano.tensor.nnet.nnet.relu
        elif activation == 'elu':
            self.activation = theano.tensor.nnet.nnet.elu
        elif activation == 'selu':
            self.activation = theano.tensor.nnet.nnet.selu
        elif activation == 'binary_crossentropy':
            self.activation = theano.tensor.nnet.nnet.binary_crossentropy
        elif activation == 'sigmoid_binary_crossentropy':
            self.activation = theano.tensor.nnet.nnet.sigmoid_binary_crossentropy
        elif activation == 'categorical_crossentropy':
            self.activation = theano.tensor.nnet.nnet.categorical_crossentropy
        elif activation == 'h_softmax':
            self.activation = theano.tensor.nnet.nnet.h_softmax
        else:
            raise ValueError('{} is not a valid activation function.'.format(activation))

        self.use_bias = use_bias
        self.units = units
        self.kernel_initializer = kernel_initializer

    def set_parameters(self, number_of_inputs):
        k_i = self.kernel_initializer
        if k_i == 'uniform':
            weights = np.random.random((number_of_inputs, self.units))
            bias = np.random.random((1,self.units))
        elif k_i == 'normal':
            weights = np.random.randn(number_of_inputs, self.units)
            bias = np.random.randn(1,self.units)
        else :
            raise ValueError("Opción no válida.")

        self.weights = theano.shared(value = weights, name = 'l{}'.format(Dense.number_of_layers))
        self.bias = theano.shared(value = bias, name = 'b{}'.format(Dense.number_of_layers), broadcastable=(True,False))

    def get_layer_weights(self):
        return self.weights

    def get_layer_bias(self):
        return self.bias

    def get_layer_symbolic_output(self, number_of_inputs):
        input_ = tensor.matrix('input_{}'.format(self.number_of_layers))
        output = self.activation(tensor.dot(self.weights,input_)) + self.bias
        return output

    
    def get_units(self):
        return self.units
