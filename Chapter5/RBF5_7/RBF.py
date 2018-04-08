"""
Author: Victoria
Created on 2017.9.24 11:30
"""
import numpy as np
import matplotlib.pyplot as plt

class RBF():
    def __init__(self):
        self.hidden_num = 4
        self.w = np.random.rand(self.hidden_num, 1)
        self.beta = np.random.rand(self.hidden_num, 1)
        self.cs = np.random.rand(self.hidden_num, 2)

    def forward(self, x):
        """
        Forward process of RBF network.
        Input:
            x: np.array with shape [1, 2]
        """
        self.x = x
        self.dist = np.sum((x - self.cs)**2, axis=1, keepdims=True)
        self.h = np.exp(-self.beta * self.dist)
        self.y = self.w.T.dot(self.h)
        return self.y[0, 0]

    def grad(self, y):
        """
        Compute gradients.
        Input:
            y: int, label.
        """
        grad_y = self.y - y
        self.grad_w = grad_y * self.h
        grad_h = grad_y * self.w
        self.grad_beta = - grad_h * self.h * self.dist
        self.grad_cs = grad_h * self.h * 2 * self.beta * (self.x - self.cs)

        self.grads = [self.grad_w, self.grad_beta, self.grad_cs]

    def update(self, lr=0.1):
        """
        Update params with gradient.
        """
        self.w -= lr * self.grad_w
        self.beta -= lr * self.grad_beta
        self.cs -= lr * self.grad_cs

    def train(self, X, y):
        """
        Training model with training set (X, y).
        Input:
            X: list of np.array with shape [1, 2]
            y: list of label.
        """
        epochs = 200
        losses = []
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                self.forward(X[i])
                self.grad(y[i])
                self.update()
            for i in range(len(X)):
                loss += self.loss(X[i], y[i])
            losses.append(loss)
        return losses


    def loss(self, x, y):
        predict = self.forward(x)
        mse = 0.5 * (predict-y)**2
        return mse

    def gradCheck(self, x, y):
        """
        Checking grad() right or not with one example.
        """
        self.forward(x)
        self.grad(y)

        epsilon = 1e-5
        params = [self.w, self.beta, self.cs]
        for k in range(len(params)):
            m, n = params[k].shape
            for i in range(m):
                for j in range(n):
                    params[k][i, j] += epsilon
                    max_loss = self.loss(x, y)
                    params[k][i, j] -= 2*epsilon
                    min_loss = self.loss(x, y)
                    num_grad = (max_loss - min_loss) / (2*epsilon)
                    params[k][i, j] += epsilon
                    ana_grad = self.grads[k][i, j]
                    #print "ana_grad: {}, num_grad: {}".format(ana_grad, num_grad)
                    if np.abs(num_grad - ana_grad) / np.abs(num_grad) > 1e-7:
                        raise Exception("grad error! {} {} {}".format(k, i, j))
        print ("grad checking successful")

if __name__=="__main__":
    #XOR data
    train_X = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
    train_y = [0, 1, 1, 0]

    #checking grads
    net = RBF()
    net.gradCheck(train_X[0], train_y[0])

    #training
    losses = net.train(train_X, train_y)
    plt.plot(range(len(losses)), losses, 'r-')
    plt.show()

    #predict
    predicts = []
    for i in range(4):
        predict = net.forward(train_X[i])
        predicts.append(predict)
    print (predicts)