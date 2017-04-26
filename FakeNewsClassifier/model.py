import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import Counter
from sklearn import model_selection
import csv


#TODO: Add dropout
class NewsClassifier(object):
    @staticmethod
    def multilayer_perceptron(input_tensor,weights,biases):
        layer_1_multiply = tf.matmul(tf.cast(input_tensor,tf.float32), weights['h1'])
        layer_1_add = tf.add(layer_1_multiply, biases['b1'])
        layer_1_activation = tf.nn.relu(layer_1_add)

        layer_2_multiply = tf.matmul(layer_1_activation, weights['h2'])
        layer_2_add = tf.add(layer_2_multiply, biases['b2'])
        layer_2_activation = tf.nn.relu(layer_2_add)

        out_layer_multiply = tf.matmul(layer_2_activation, weights['out'])
        out_layer_add = tf.add(out_layer_multiply, biases['out'])

        return out_layer_add
