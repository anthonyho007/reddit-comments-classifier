from sklearn.preprocessing import Binarizer
from sklearn.utils.validation import check_X_y, check_array
import pandas as pd
import numpy as np

class BernoulliNB:
    def __init__(self):
        self.prob_class = None
        self.prob_x_c = None
        self.prob_x_cp = None

    def fit(self, X, y):
        X = self.binarize(X)
        classes = np.unique(y)
        probs_class = []
        probs_x_c = []
        probs_x_cp = []

        for c in classes:
            prob_class, prob_x_c, prob_x_cp = self.populate_class_stats(X, y, c)
            probs_class.append(prob_class)
            probs_x_c.append(prob_x_c)
            probs_x_cp.append(prob_x_cp)

        self.prob_class = np.asarray(probs_class)
        self.prob_x_c = np.asarray(probs_x_c)
        self.prob_x_cp = np.asarray(probs_x_cp)

    def binarize(self, X):
        binarizer = Binarizer().fit(X)
        return binarizer.transform(X)

    def populate_class_stats(self, X, y, c):
        X, y = check_X_y(X, y, accept_sparse="csr")
        Xc = X[y == c, :]
        Xcp = X[y != c, :]

        # P(c)
        prob_class = Xc.shape[0] / X.shape[0]

        # number of sample where x is in class c
        Nxc = np.asarray(Xc.sum(axis=0))

        # number of sample where x not in class c
        Nxcp = np.asarray(Xcp.sum(axis=0))

        n_feat = Nxc.shape[1]

        # number of samples in class c
        Nsc = Xc.shape[0]

        # number of samples not in class c
        Nscp = Xcp.shape[0]

        # P(x | c) with lapace smoothing
        prob_x_c = (Nxc + 1) / (Nsc + 2)

        # P(x | c') with lapace smoothing
        prob_x_cp = (Nxcp + 1) / (Nscp  + 2)

        return prob_class, prob_x_c.reshape(n_feat), prob_x_cp.reshape(n_feat)

    def predict(self, X):
        X = check_array(X, accept_sparse="csr")
        X = self.binarize(X)
        one = np.ones(X.shape[1])
        l1 = np.log(self.prob_x_c/self.prob_x_cp).T
        l2 = np.log((1-self.prob_x_c)/(1-self.prob_x_cp)).T

        pred = np.log(self.prob_class/(1-self.prob_class)) + \
                + (X @ l1 + (one - X) @ l2)
        return np.argmax(pred, axis=1)
