import tensorflow as tf
import numpy as np
import random

"""
모든 n차 회귀를 계산할수있게끔 설계한 all_regression 함수
degree(차수)에 따라 각 x의 차수항에 대입되는 variable을 만들고, 계산하게끔 설계
(좀 더 컴공적 설계)
"""


class all_regression:

    def __init__(self, degree, learning_rate):
        self.Degree = degree
        self.optimizer = tf.optimizers.Adam(lr=learning_rate)
        self.variables = self.make_variable(degree)

    def input_data(self, x, y):
        self.X_data = x
        self.Y_data = y
        self.variables = self.make_variable(self.Degree)

    def make_variable(self, Degree):
        new_variable = []
        for i in range(Degree):
            new_variable.append(tf.Variable(random.random()))
        new_variable.append(tf.Variable(random.random()))
        return new_variable

    def compute_loss(self):
        y_predicted = 0
        for i in range(self.Degree + 1):
            y_predicted += self.variables[i] * (self.X_data ** (self.Degree - i))
        loss = tf.reduce_mean((self.Y_data - y_predicted) ** 2)
        return loss

    def training(self, training_cycle):
        for i in range(training_cycle):
            self.optimizer.minimize(self.compute_loss, var_list=self.variables)

    def return_value(self):
        line_x = np.arange(min(self.X_data), max(self.X_data), 0.1)
        line_y = 0
        for i in range(self.Degree + 1):
            x_value = line_x ** (self.Degree - i)
            line_y += self.variables[i] * x_value

        return line_x, line_y