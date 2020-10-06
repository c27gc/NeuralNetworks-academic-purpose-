import numpy as np
import pandas as pd
import theano.tensor as tensor
import theano
import orion.optimizers as optimizers

#theano.config.optimizer='None'

class Sequential():
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def symbolic_feedfoward(self, input_, number_of_inputs):
        weights = []
        bias = []
        for layer_number in range(len(self.layers)):
            if layer_number == 0:
                model = input_
                n_inputs = number_of_inputs
            else:
                self.layers[layer_number].set_parameters(n_inputs)
                weights.append(self.layers[layer_number].get_layer_weights())
                bias.append(self.layers[layer_number].get_layer_bias())
                #output = self.layers[layer_number].get_layer_symbolic_output()
                n_inputs = self.layers[layer_number].get_units()
                model = self.layers[layer_number].activation(theano.dot(model, weights[-1]) + bias[-1])

        return model, weights, bias

    def symbolic_feedfoward_val(self, input_, weights, bias):
        model = input_
        for n_layer in range(len(weights)):
            model = self.layers[n_layer + 1].activation(theano.dot(model, weights[n_layer]) + bias[n_layer])
        return model

    #def stochastic_gradient_descent(self):
        #self.dweights = [[w, w - self.learning_rate * tensor.grad(self.loss_, w)] for w in self.weights]
        #self.dbias = [[b, b - self.learning_rate * tensor.grad(self.loss_, b)] for b in self.bias]
        #update = self.dweights + self.dbias

        #return update

    def compile(self, loss = 'mse', optimizer = 'sgd'):
        self.learning_rate = 0.01
        number_of_inputs = self.layers[0].number_of_inputs

        self.x_train_ = tensor.matrix('x_train_')
        self.y_train_ = tensor.matrix('y_train_')

        self.estimation_train, self.weights, self.bias = self.symbolic_feedfoward(self.x_train_, number_of_inputs)

        self.x_val_ = tensor.matrix('x_val_')
        self.y_val_ = tensor.matrix('y_val_')

        self.estimation_val = self.symbolic_feedfoward_val(self.x_val_, self.weights, self.bias)

        if loss == 'mse':
            self.loss_ = (1/tensor.shape(self.y_train_)[0])*tensor.sum(tensor.sqr((self.y_train_ - self.estimation_train)))
            self.loss_val_ = (1/tensor.shape(self.y_val_)[0])*tensor.sum((self.y_val_ - self.estimation_val)**2)
        else:
            raise ValueError("Opción no válida.")
        
        if optimizer == 'sgd':
            self.optimizer = optimizers.stochastic_gradient_descent(self.loss_, self.weights + self.bias, lr = self.learning_rate)
        elif optimizer == 'adam':
            self.optimizer = optimizers.adam(self.loss_, self.weights + self.bias)
        else:
            raise ValueError("Opción no válida.")

        self.train = theano.function(inputs = [self.x_train_,self.y_train_, self.x_val_, self.y_val_],\
                    outputs = [ self.loss_, self.loss_val_], updates = self.optimizer() )

    def fit(self, x_train, y_train, x_val, y_val, epochs):
        loss = []
        loss_val = []

        for iter_ in range(epochs):
            loss_, loss_val_ = self.train(x_train, y_train, x_val, y_val)
            print("Epoch {}/{}\nloss: {} - val_loss: {}".format(iter_+1, epochs,float(loss_), float(loss_val_)))
            
            if float(loss_) <= 0.01 and float(loss_val_) <= 0.01:
                break
            
            loss.append(float(loss_))
            loss_val.append(float(loss_val_))

        #print(loss)

    def predict(self):
        pass

