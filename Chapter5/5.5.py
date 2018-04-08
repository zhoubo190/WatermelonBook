"""
Author: Victoria
Created on 2017.9.22 9:00
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    f = 1.0 / (1+np.exp(-x))
    return f

def softmax(x):
    #array
    if len(x.shape)==2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        f = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    #vector
    else:
        x = x - np.max(x)
        exp_x = np.exp(x)
        f = exp_x / np.sum(exp_x)
    return f

class Net():
    def __init__(self):
        self.hidden_num = 4
        self.output_num = 2
        self.W1 = np.random.rand(2, self.hidden_num)
        self.b1 = np.zeros((1, self.hidden_num))
        self.W2 = np.random.rand(self.hidden_num, self.output_num)
        self.b2 = np.zeros((1, self.output_num))

    def forward(self, X):
        """
        Forward process of this simple network.
        Input:
            X: np.array with shape [N, 2]
        """
        self.X = X
        self.z1 = X.dot(self.W1) + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = self.h.dot(self.W2) + self.b2
        self.y = softmax(self.z2)

    def grad(self, Y):
        """
        Compute gradient of parameters for training data (X, Y). X is saved in self.X.
        """
        grad_z2 = self.y - Y
        self.grad_W2 = self.h.T.dot(grad_z2)
        self.grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        grad_h = grad_z2.dot(self.W2.T)
        grad_z1 = grad_h * self.h * (1-self.h)
        self.grad_W1 = self.X.T.dot(grad_z1)
        self.grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
        #used for grad_check()
        self.grads = [self.grad_W1, self.grad_W2, self.grad_b1, self.grad_b2]

    def update(self, lr=0.1):
        """
        Update parameters with gradients.
        """
        self.W1 -= lr*self.grad_W1
        self.b1 -= lr*self.grad_b1
        self.W2 -= lr*self.grad_W2
        self.b2 -= lr*self.grad_b2

    def bp(self, Y):
        """
        BP algorithm on data (X, Y)
        Input:
            Y: np.array with shape [N, 2]
        """
        self.grad(Y)
        self.update()

    def loss(self, X, Y):
        """
        Compute loss on X with current model.
        Input:
            X: np.array with shape [N, 2]
            Y: np.array with shape [N, 2]
        Return:
            cost: float
        """
        self.forward(X)
        cost = np.sum(-np.log(self.y) * Y)
        return cost

    def grad_check(self, X, Y):
        """
        Check gradient in bp() is right or not.
        Input:
            X:
            Y:
        """
        self.forward(X)
        self.grad(Y)
        #W1, W2, b1, b2
        epsilon = 1e-5
        params = [self. W1, self.W2, self.b1, self.b2]

        for k in range(len(params)):
            m, n = params[k].shape
            param = params[k]
            for i in range(m):
                for j in range(n):
                    param[i, j] = param[i, j] + epsilon
                    max_loss = self.loss(X, Y)
                    param[i, j] = param[i, j] - 2*epsilon
                    min_loss = self.loss(X, Y)
                    num_grad = (max_loss - min_loss) / (2*epsilon)
                    ana_grad = self.grads[k][i, j]
                    if np.abs( num_grad-ana_grad ) > 1e-5:
                        print ("grad error! {} {} {}".format(k, i, j))
        print ("grad checking successful")

if __name__=="__main__":

    #read data from xls file
    train_X = []
    train_Y = []
    book = pd.read_csv("../data/watermelon_3a.csv")
    X1 = book.values[:, 1]
    X2 = book.values[:, 2]
    y = book.values[:, 3]
    N = len(X1)
    for i in range(N):
        train_X.append(np.array([X1[i], X2[i]]))
        train_y = np.array([0, 0])
        train_y[int(y[i])] = 1
        train_Y.append(train_y)

    #check grads
    net = Net()
    net.grad_check(train_X[0].reshape(1, 2), train_Y[0].reshape(1, 2))

    #training
    iterations = 10000
    train_Xs = np.vstack(train_X)
    train_Ys = np.vstack(train_Y)
    #training network with standard BP
    standard_net = Net()
    train_losses = []
    for iter in range(iterations):
        for i in range(N):
            standard_net.forward(train_X[i].reshape(1, 2))
            standard_net.bp(train_Y[i].reshape(1, 2))
        loss = standard_net.loss(train_Xs, train_Ys)
        train_losses.append(loss)
    line1, = plt.plot(range(iterations), train_losses, "r-")
    #training network with accumulated BP
    accumulated_net = Net()
    train_losses = []
    for iter in range(iterations):
        accumulated_net.forward(train_Xs)
        accumulated_net.bp(train_Ys)
        loss = accumulated_net.loss(train_Xs, train_Ys)
        train_losses.append(loss)
    line2, = plt.plot(range(iterations), train_losses, "b-")
    plt.legend([line1, line2], ["BP", "Accumulated BP"])
    plt.show()