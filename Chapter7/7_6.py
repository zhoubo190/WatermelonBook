# -*-coding:gbk -*-
"""
@Author: Victoria
@Date: 2017.10.19 21:30
"""
import pandas as pd

class AODE():
    def __init__(self, d, class_num = 2):
        #discrete features number
        self.d = d
        self.class_num = class_num

    def train(self, X, y):
        """
        The training process of AODE is to save estimated joint probability.
        """
        count_xj_c_xi = {}
        count_c_xi = {}
        prob_xj_c_xi = {}
        prob_c_xi = {}

        N = len(X)

        attrs = []
        for i in range(self.d):
            attr = []
            for n in range(N):
                if X[n][i] not in attr:
                    attr.append(X[n][i])
            attrs.append(attr)

        for c in range(self.class_num):
            count_c_xi[c] = {}
            prob_c_xi[c] = {}
            count_xj_c_xi[c] = {}
            prob_xj_c_xi[c] = {}

            for i in range(self.d):
                count_c_xi[c][i] = {}
                prob_c_xi[c][i] = {}
                count_xj_c_xi[c][i] = {}
                prob_xj_c_xi[c][i] = {}
                for attr_i in attrs[i]:
                    count_c_xi[c][i][attr_i] = 0
                    prob_c_xi[c][i][attr_i] = 0
                    count_xj_c_xi[c][i][attr_i] = {}
                    prob_xj_c_xi[c][i][attr_i] = {}
                    for j in range(self.d):
                        count_xj_c_xi[c][i][attr_i][j] = {}
                        prob_xj_c_xi[c][i][attr_i][j] = {}
                        for attr_j in attrs[j]:
                            count_xj_c_xi[c][i][attr_i][j][attr_j] = 0
                            prob_xj_c_xi[c][i][attr_i][j][attr_j] = 0


        for n in range(N):
            for i in range(self.d):
                    count_c_xi[y[n]][i][X[n][i]] += 1

                    for j in range(self.d):
                        count_xj_c_xi[y[n]][i][X[n][i]][j][X[n][j]] += 1


        for c in range(self.class_num):
            for i in range(self.d):
                #the values number of i-th attribution
                v_i = len(attrs[i])
                for attr_i_value, N_c_xi in count_c_xi[c][i].items():
                    prob_c_xi[c][i][attr_i_value] = float(N_c_xi + 1) / (N + self.class_num *v_i)

                    for j in range(self.d):
                        v_j = len(attrs[j])
                        for attr_j_value, N_c_xi_xj in count_xj_c_xi[c][i][attr_i_value][j].items():
                            prob_xj_c_xi[c][i][attr_i_value][j][attr_j_value] = float(N_c_xi_xj + 1) / (N_c_xi + v_j)

        self.count_xj_c_xi = count_xj_c_xi
        self.count_c_xi = count_c_xi
        self.prob_xj_c_xi = prob_xj_c_xi
        self.prob_c_xi = prob_c_xi

    def predict(self, x):
        probs = []
        for c in range(self.class_num):
            prob_c = 0
            for i in range(self.d):

                prob_j_c_i_product = 1.0
                for j in range(self.d):
                    prob_j_c_i_product *= self.prob_xj_c_xi[c][i][x[i]][j][x[j]]
                prob_c_i_term = self.prob_c_xi[c][i][x[i]] * prob_j_c_i_product
            prob_c += prob_c_i_term
            probs.append(prob_c)
        label = probs.index(max(probs))
        prob = max(probs)
        return label, prob



if __name__=="__main__":
    workbook = pd.read_csv("../data/watermelon_3.csv", encoding="gb18030")
    X = workbook.values[:, 1:7]
    y = workbook.values[:, 9]
    for i in range(len(y)):
        if y[i] == "是":
            y[i] = 1
        else:
            y[i] = 0
    aode = AODE(d=6)
    aode.train(X, y)
    label, prob = aode.predict(["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"])
    print ("the predict label is {} with prob {}".format(label, prob))