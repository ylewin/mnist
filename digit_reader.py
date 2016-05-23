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
        self.biases = [np.random.rand(1, out_dim) * 2 - 1 for out_dim in dims[1:]]
        self.weights = [np.random.rand(in_dim, out_dim) * 2 - 1 \
            for in_dim, out_dim in zip(dims[:-1], dims[1:])]
        
        
        
    def feedforward(self, x):
        
        # If x is one dimensional array might couse
        # problems with the dot product, numpy is trying to
        # do vector dot insted of matrix dot
        
        a = deepcopy(x)
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(a, w) + b)
        return a
    
        
    def updateNetwork(self, data, learn_rate=0.001):
        """ ''data'' is list of tuples (x, y) where x is
        the input and y is the wanted output"""
         
        for x, y in data:
            nabla_w, nabla_b = self.backprop(x, y)
            self.weights -= learn_rate * nabla_w
            self.biases -= learn_rate * nabla_b        
        
        
    
    def backprop(self, x, y):
        nabla_w = np.zeros(self.weights)
        nabla_b = np.zeros(self.biases)
        
        # Calc the derivetive        
        
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
        
        return 1.0 / x.shape[1] * (tx - x)
        
        
        

nn = Network([28**2, 100, 10])

def predict_digit(d):
    out = nn.feedforward(d.reshape(1, 28**2))
    m  = out.max()
    for i in range(out.shape[1]):
        if out[0][i] == m:
            return i
            
def evaluate_network(data=mnist.read_mnist_files()):
    curr = 0.0
    for i in range(len(data[0])):
        if predict_digit(data[0][i]) == data[1][i]:
            curr += 1.0
    return curr / len(data[0])
    
        




















