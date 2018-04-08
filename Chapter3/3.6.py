"""
Author: Victoria
Created on: 2017.9.15 11:00
"""
import numpy as np
import matplotlib.pyplot as plt

def readData():
    """
    Read data from txt file.
    Return:
        X1, y1, X2, y2, X3, y3: X is list with shape [50, 4],
                                y is list with shape [50,]
    """
    X1 = []
    y1 = []
    X2 = []
    y2 = []
    X3 = []
    y3 = []
    #read data from txt file
    with open("../data/bezdekIris.txt", "r") as f:
        for line in f:
            x = []
            iris = line.strip().split(",")
            for attr in iris[0:4]:
                x.append(float(attr))

            if iris[4]=="Iris-setosa":
                X1.append(x)
                y1.append(1)
            elif iris[4]=="Iris-versicolor":
                X2.append(x)
                y2.append(2)
            else:
                X3.append(x)
                y3.append(3)
    return X1, y1, X2, y2, X3, y3

def tenFoldData(X1, X2):
    """
    Generate 10-fold training data. Each fold includes 5 positive and 5 negtive.
    Input:
        X1: list with shape [50, 4]. Instances in X1 belong to positive class.
        X2: list with shape [50, 4]. Instances in X2 belong to negtive class.
    Return:
        folds: list with shape [10, 10, 4].
        y: list with shape [10, 10]
    """
    print (len(X1))
    print (len(X2))
    folds = []
    y = []
    for i in range(10):
        fold = []
        fold += X1[ i*5:(i+1)*5 ]
        fold += X2[ i*5:(i+1)*5 ]
        folds.append(fold)
        y.append([1]*5 + [0]*5)
    return folds, y

def LR(X, y):
    """
    Given training dataset, return optimal params of LR algorithm with Newton method.
    Input:
        X: np.array with shape [N, d]. Input.
        y: np.array with shape [N, 1]. Label.
    Return:
        beta: np.array with shape [1, d]. Optimal params with Newton method
    """
    N, d = X.shape
    lr = 0.001
    #initialization
    beta = np.ones((1, d)) * 0.1
    #shape [N, 1]
    z = X.dot(beta.T)

    for i in range(150):
        #shape [N, 1]
        p1 = np.exp(z) / (1 + np.exp(z))
        #shape [N, N]
        p = np.diag((p1 * (1-p1)).reshape(N))
        #shape [1, d]
        first_order = -np.sum(X * (y - p1), 0, keepdims=True)

        #update
        beta -= first_order * lr
        z = X.dot(beta.T)

    l = np.sum(y*z + np.log( 1+np.exp(z) ) )
    #print l
    return beta


def testing(beta, X, y):
    """
    Given trained LR model, return error number in input X.
    Input:
        beta: np.array with shape [1, d]. params of LR model
        X: np.array with shape [N, d]. Testing instances.
        y: np.array with shape [N, 1]. Testing labels.
    Return:
        error_num: Error num of LR on X.
    """
    predicts = ( X.dot(beta.T) >= 0 )
    error_num = np.sum(predicts != y)
    return error_num

def tenFoldCrossValidation(folds, y):
    """
    Return erroe num of 10-fold cross validation.
    Input:
        folds: list with shape [10, 10, 4].
        y: list with shape [10, 10]
    Return:
        ten_fold_error_nums:
    """
    ten_fold_error_nums = 0
    for i in range(10):
        train_X = folds[:i] + folds[i+1:]
        train_y = y[:i] + y[i+1:]
        val_X = folds[i]
        val_y = y[i]
        train_X = np.array(train_X).reshape(-1, 4)
        train_y = np.array(train_y).reshape([-1, 1])
        val_X = np.array(val_X)
        val_y = np.array(val_y).reshape([-1, 1])
        beta = LR(train_X, train_y)
        error_num = testing(beta, val_X, val_y)
        ten_fold_error_nums += error_num
    return ten_fold_error_nums

def LOO(X, y):
    """
    Return erroe num of LOO.
    Input:
        X: list with shape [100, 4].
        y: list with shape [100]
    Return:
        loo_error_nums:
    """
    loo_error_nums = 0
    for i in range(100):
        train_X = X[:i] + X[i+1:]
        train_y = y[:i] + y[i+1:]
        val_X = X[i]
        val_y = y[i]
        train_X = np.array(train_X).reshape(-1, 4)
        train_y = np.array(train_y).reshape([-1, 1])
        val_X = np.array(val_X)
        val_y = np.array(val_y).reshape([-1, 1])
        beta = LR(train_X, train_y)
        error_num = testing(beta, val_X, val_y)
        loo_error_nums += error_num
    return loo_error_nums

if __name__=="__main__":
    #data read from txt file
    X1, y1, X2, y2, X3, y3 = readData()

    #10-fold cross validation
    print ("10-fold cross validation...")
    #X1 and X2
    folds, y = tenFoldData(X1, X2)
    round1_ten_fold_error_nums = tenFoldCrossValidation(folds, y)
    #X1, X3
    folds, y = tenFoldData(X1, X3)
    round2_ten_fold_error_nums = tenFoldCrossValidation(folds, y)
    #X2, X3
    folds, y = tenFoldData(X3, X2)
    round3_ten_fold_error_nums = tenFoldCrossValidation(folds, y)
    ten_fold_error_nums = round1_ten_fold_error_nums + round2_ten_fold_error_nums \
                          + round3_ten_fold_error_nums

    #LOO
    print ("LOO ...")
    #X1, X2
    X = X1 + X2
    y = [1]*len(X1) + [0]*len(X2)
    round1_loo_error_nums = LOO(X, y)
    #X1, X3
    X = X1 + X3
    y = [1]*len(X1) + [0]*len(X2)
    round2_loo_error_nums = LOO(X, y)
    #X2, X3
    X = X3 + X2
    y = [1]*len(X1) + [0]*len(X2)
    round3_loo_error_nums = LOO(X, y)
    loo_error_nums = round1_loo_error_nums + round2_loo_error_nums \
                     + round3_loo_error_nums
    print (round1_ten_fold_error_nums, round2_ten_fold_error_nums, round3_ten_fold_error_nums)
    print ("10-fold cross validation error num: {}/300".format(ten_fold_error_nums))
    print (round1_loo_error_nums, round2_loo_error_nums, round3_loo_error_nums)
    print ("LOO error num: {}/300".format(loo_error_nums))