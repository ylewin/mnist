# -*- coding: utf-8 -*-

import mnist_loader as mnist
import numpy as np
from copy import deepcopy

DEBUG = False


class Network(object):
    
    def __init__(self, dims):
        
        if DEBUG:
            np.random.seed(0)
            
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.biases = [np.random.rand(1, out_dim) * 2 - 1 for out_dim in dims[1:]]
        self.weights = [np.random.rand(in_dim, out_dim) * 2 - 1 \
            for in_dim, out_dim in zip(dims[:-1], dims[1:])]
        
        
        
    def feedforward(self, x):
        
        # If x is one dimensional array might couse
        # problems with the dot product, numpy is trying to
        # do vector dot insted of matrix dot
            
        activations = [deepcopy(x)]
        zs = []
        
        for w, b in zip(self.weights, self.biases):
            zs.append(np.dot(activations[-1], w) + b)
            activations.append(self.sigmoid(zs[-1]))
        return activations, zs
    
        
    def updateNetwork(self, data, learn_rate=0.001):
        """ ''data'' is list of tuples (x, y) where x is
        the input and y is the wanted output"""
        
        errors = []
        error = []
         
        for i, (x, y) in enumerate(data):
            activations, zs = self.feedforward(x)
            nabla_w, nabla_b = self.network_deriv(activations, zs, y)
            for l in range(self.num_layers):
                self.weights[l] -= learn_rate * nabla_w[l]
                self.biases[l] -= learn_rate * nabla_b[l]
            error.append(self.costfunc(activations[-1], y))
            if i % 5000 == 0 and i != 0:
                errors.append(sum(error) / len(error))
                if len(errors) > 1:
                    print("mean error: ", errors[-1], "      delta error: ", errors[-1] - errors[-2])
                else:
                    print("mean error: ", errors[-1])
                error = []
            
        
        
    
    def network_deriv(self, activations, zs, y):
        nabla_w = []
        nabla_b = []
        
        for i in range(self.num_layers):
            nabla_w.append(np.zeros(self.weights[i].shape))
            nabla_b.append(np.zeros(self.biases[i].shape))
        
        # Calc the derivetive using backpropagation
        deltas = [self.costfunc_deriv(activations[-1], y) * self.sigmoid_deriv(zs[-1])]
        
        
        #Consider using nagative index insted of reversing
        for layer in reversed(range(self.num_layers - 1)):
            deltas.append(np.dot( deltas[-1], self.weights[layer + 1].T) * self.sigmoid_deriv(zs[layer]))
        
        for i, delta in enumerate(reversed(deltas)):            
            nabla_b[i] = delta
            nabla_w[i] = delta * activations[i].T
        
        return nabla_w, nabla_b
        
        
    def sigmoid(self, x):
            return 1.0 / (1.0 + np.exp(x))
        
        
    def sigmoid_deriv(self, x):
        ex = np.exp(x)
        return -ex / (ex + 1.0)**2
        
    def costfunc(self, x, tx):
        assert(x.shape[0] == tx.shape[0] == 1)
        assert(x.shape == tx.shape)
        
        return 0.5 * ((tx - x)**2).mean()
        
    def costfunc_deriv(self, x, tx):
        assert(x.shape[0] == tx.shape[0] == 1)
        assert(x.shape == tx.shape)
        
        # assuming x is 1D so x.shape[1] is the length of x
        return -1.0 / x.shape[1] * (tx - x)
        
        
        

nn = Network([28**2, 300, 10])

def predict_digit(d):
    out = nn.feedforward(d.reshape(1, -1))[0][-1]
    m  = out.max()
    for i in range(out.shape[1]):
        if out[0][i] == m:
            return i       

def evaluate_network(data):
    """ ''data'' is list of tuples (x, y) where x is
        the input and y is the real output"""

    curr = 0.0
    for x, y in data:
        if predict_digit(x) == arr_to_digit(y):
            curr += 1.0
    return curr / len(data)
    
        
def digit_to_arr(d):
    rv = np.zeros((1,10))
    rv[0][d] = 1
    return rv

def arr_to_digit(arr):
   return arr.argmax()
            
def create_data(mnist_data): 
    return [(x.reshape(1, -1) / 255, digit_to_arr(y)) for x, y in \
        zip(mnist_data[0], mnist_data[1])]
        
def create_test_data(mnist_data): 
    return [(x.reshape(1, -1) / 255, digit_to_arr(y)) for x, y in \
        zip(mnist_data[2], mnist_data[3])]



def train(data, times, learn_rate = 0.001):
    for i in range(times):
        nn.updateNetwork(data, learn_rate)

def print_num(data, idx):
    mnist.plot(data[idx][0].reshape((28, 28)))
    print(arr_to_digit(nn.feedforward(data[idx][0])[0][-1]))    
    














