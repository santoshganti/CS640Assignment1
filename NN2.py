#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:25:57 2019

@author: Santosh,Sahil
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def sigmoid_derivative(s):
    return s * (1 - s)


def softmax(z):
    exp_z = np.exp(z)
    softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return softmax_scores


def softmax_to_y(softmax_scores):
    return np.argmax(softmax_scores, axis=1)


def reLU(x):
    return x * (x > 0)


def reLU_derivative(x):
    return 1. * (x > 0)


def error_rate(y_pred, y_test):
    m = y_pred.shape[0]
    return np.sum(1 - (y_pred == y_test)) / m


def plot_decision_boundary(model, X, y):
    x1_array, x2_array = np.meshgrid(np.arange(-10, 10, 0.01), np.arange(-10, 10, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


class NN(object):

    def __init__(self, input_dimension, output_dimension, nodes, alpha=0.1, num_epochs=1000):
        # weights
        self.input_weight = np.random.randn(input_dimension, nodes) / np.sqrt(input_dimension)
        self.hidden_weight = np.random.randn(nodes, output_dimension) / np.sqrt(nodes)

        # bias
        self.input_bias = np.zeros((1, nodes))
        self.output_bias = np.zeros((1, output_dimension))
        self.alpha = alpha
        self.epochs = num_epochs

    def hyperparameters(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs

    def forward_propagation(self, X):
        # dot product of X (input) and first set
        self.hidden = reLU(np.dot(X, self.input_weight) + self.input_bias)
        # dot product of hidden layer and second set
        self.output = softmax(np.dot(self.hidden, self.hidden_weight) + self.output_bias)
        return self.output

    def backward_propagation(self, X, y):
        d = X.shape[0]
        one_hot_y = np.zeros_like(self.output)
        for i in range(y.shape[0]):
            one_hot_y[i, y[i]] = 1

        self.o_error = self.output - one_hot_
        self.o_delta = self.o_error

        # error: how much hidden layer weights contributed to output error
        self.hid_error = self.o_delta.dot(self.hidden_weight.T)

        # applying derivative of reLu to hidden error
        self.hid_delta = self.hid_error * reLU_derivative(self.hidden)

        w2 = self.hidden.T.dot( self.o_delta) / d
        b2 = np.sum(self.o_delta, axis=0, keepdims=True) / d
        w1 = X.T.dot( self.hid_delta) / d
        b1 = np.sum(self.hid_delta, axis=0, keepdims=True) / d

        # Return updated gradients
        values = { "w1": w1,
                    "b1": b1,
                   "w2": w2,
                "b2": b2}
        return values

    # Updates the weights after calculating gradient in the self propagation step
    def update_weight(self, grads):
        self.input_bias -= self.alpha * grads["b1"]
        self.output_bias -= self.alpha * grads["b2"]
        self.input_weight -= self.alpha * grads["w1"]
        self.hidden_weight -= self.alpha * grads["w2"]

    # Computes the cross entropy between the actual and predicted values
    def compute_cost(self, o_z, y):
        m = y.shape[0]
        log_likelihood = -np.log(o_z[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    # Fits the neural network using the training dataset
    # Returns the training error for every 10th epoch
    def fit(self, X_train, y_train):
        train_error = [0.5]
        for i in range(self.epochs):
            output = self.forward_propagation(X_train)
            grads = self.backward_propagation(X=X_train, y=y_train)
            self.update_weight(grads)
            if (i % 10 == 0):
                # hot_y = softmax_to_y(output)
                train_error += [self.compute_cost(output, y_train)]
        return train_error

    # Fits the neural network using the training dataset,
    # calculates train as well as test error rate alongside
    def fit_test_train(self, X_train, y_train, X_test, y_test):
        train_error = []
        test_error = []
        for i in range(self.epochs):
            output_train = self.forward_propagation(X_train)
            grads = self.backward_propagation(X=X_train, y=y_train)
            self.update_weight(grads)
            if (i % 10 == 0):
                train_error += [self.compute_cost(output_train, y_train)]
                output_test = self.forward_propagation(X_test)
                test_error += [self.compute_cost(output_test, y_test)]
        error = [None] * 2
        error[0] = train_error
        error[1] = test_error
        return error

    def predict(self, X_test):
        output = self.forward_propagation(X_test)
        return softmax_to_y(output)


def k_fold(X, y, k, nn):
    X = np.array_split(X, k)
    y = np.array_split(y, k)

    test_error = [None] * k

    for i in range(k):
        b = 0
        for j in range(k):
            if j == i:
                X_test = X[i]
                y_test = y[i]
            else:
                if b == 0:
                    X_train = X[j]
                    y_train = y[j]
                    b += 1
                else:
                    X_train = np.concatenate([X_train, X[i]])
                    y_train = np.concatenate([y_train, y[i]])
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        test_error[i] = error_rate(y_pred, y_test)
    return np.sum(test_error) / (k + 1)


class regularizationL2(NN):

    def __init__(self, input_dimension, nodes, output_dimension, alpha=0.5, num_epochs=1000, reg_para=0.001):
        self.reg = reg_para
        super().__init__(input_dimension, nodes, output_dimension, alpha, num_epochs)

    def hyperparameter(self, alpha, num_epochs, reg_para):
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.reg = reg_para

    def update_weight(self, grads):
        # Adding derivative of regularization term
        self.input_bias -= self.reg * self.input_bias
        self.output_bias -= self.reg * self.output_bias
        self.input_weight -= self.reg * self.input_weight
        self.hidden_weight -= self.reg * self.hidden_weight

        self.input_bias -= self.alpha * grads["b1"]
        self.output_bias -= self.alpha * grads["b2"]
        self.input_weight -= self.alpha * grads["w1"]
        self.hidden_weight -= self.alpha * grads["w2"]


def linear():
    X = np.genfromtxt('DATA/data_linearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_lineary.csv', delimiter=',').astype(np.int64)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 10

    nn = NN(input_dim, nodes, output_dim, alpha=0.05, num_epochs=500)
    train_err = nn.fit(X, y)

    y_pred = nn.predict(X_test)

    err = error_rate(y_pred, y_test)
    print("Error in test set is ", err * 100, "%")

    plt.plot(train_err)
    plt.title("Cross Entropy with respect to Epochs")
    plt.xlabel("Number of Epochs (factor of 10)")
    plt.ylabel("Cross Entropy")
    plt.show()

    # Even though the cross entropy is not minimized,
    # our system is able to distinguish the red points from the blue points easily
    # Plotting decision boundary
    plot_decision_boundary(nn, X_test, y_test)

    # Confusion matrix
    print("Confusion Matrix \n")
    print(confusion_matrix(y_pred, y_test))
    print(classification_report(y_pred, y_test))

    X = np.genfromtxt('DATA/data_linearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_lineary.csv', delimiter=',').astype(np.int64)
    test_error = k_fold(X, y, 5, nn)
    print("Average test error is :", test_error)


def non_linear():
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 30

    nn = NN(input_dim, nodes, output_dim, alpha=0.5, num_epochs=2500)
    train_err = nn.fit(X, y)

    y_pred = nn.predict(X_test)

    err = error_rate(y_pred, y_test)
    print("Error in test set is ", err * 100, "%")

    plt.plot(train_err)
    plt.title("Cross Entropy with respect to Epochs")
    plt.xlabel("Number of Epochs (factor of 10)")
    plt.ylabel("Cross Entropy")
    plt.show()

    # Even though the cross entropy is not minimized,
    # our system is able to distinguish the red points from the blue points easily
    # Plotting decision boundary
    plot_decision_boundary(nn, X_test, y_test)

    # Confusion matrix
    print("Confusion Matrix \n")
    print(confusion_matrix(y_pred, y_test))
    print(classification_report(y_pred, y_test))

    # Cross Validation score for the linear dataset
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    test_error = k_fold(X, y, 5, nn)
    print("Average test error is :", test_error)

    # Script for checking the test and train error
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)

    nodes = 30

    nn = NN(input_dim, nodes, output_dim, alpha=0.05, num_epochs=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    err = nn.fit_test_train(X_train, y_train, X_test, y_test)

    train_err = err[0]
    test_err = err[1]

    plt.plot(train_err)
    plt.plot(test_err)
    plt.legend(('Train', 'Test'))
    plt.show()


def l2Regularization():
    X = np.genfromtxt('DATA/data_nonlinearX.csv', delimiter=',')
    y = np.genfromtxt('DATA/data_nonlineary.csv', delimiter=',').astype(np.int64)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 30
    nn = regularizationL2(input_dim, nodes, output_dim, alpha=0.1, num_epochs=2500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    err = nn.fit_test_train(X_train, y_train, X_test, y_test)

    train_err = err[0]
    test_err = err[1]

    plt.plot(train_err)
    plt.plot(test_err)
    plt.show()


def digitTraining():
    X = np.genfromtxt('DATA/Digit_X_train.csv', delimiter=',')
    y = np.genfromtxt('DATA/Digit_y_train.csv', delimiter=',').astype(np.int64)
    input_dim = int(X.shape[1])
    output_dim = int(y.max() + 1)
    nodes = 30

    nn = NN(input_dim, nodes, output_dim, alpha=0.05, num_epochs=2500)
    train_err = nn.fit(X, y)
    train_err.pop(0)

    X_test = np.genfromtxt('DATA/Digit_X_test.csv', delimiter=',')
    y_test = np.genfromtxt('DATA/Digit_y_test.csv', delimiter=',').astype(np.int64)

    y_pred = nn.predict(X_test)

    err = error_rate(y_pred, y_test)
    print(err)

    plt.plot(train_err)
    # plt.plot(test_err)
    plt.show()

    nn = regularizationL2(input_dim, nodes, output_dim, alpha=0.1, num_epochs=2500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    err = nn.fit_test_train(X_train, y_train, X_test, y_test)

    train_err = err[0]
    test_err = err[1]
    plt.plot(train_err)
    plt.plot(test_err)
    plt.show()


def main():
    print('Non-linear:\n')
    non_linear()
    print('Linear:\n')
    linear()
    print('Regularized:\n')
    l2Regularization()
    print('Digit:\n')
    digitTraining()


if __name__ == "__main__":
    main()
