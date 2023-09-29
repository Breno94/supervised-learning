import numpy as np
from scipy.stats import multivariate_normal as mvn


class NaiveBayes():
    def fit(self, X, y, smoothing=1e-2):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(y)
        for c in labels:
            current_x = X[y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing
            }
            self.priors[c] = float(len(y[y == c])) / len(y) # P(y=label)


    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y)



class Bayes():
    def fit(self, X, y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(y)
        for c in labels:
            current_x = X[y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D) * smoothing
            }
            self.priors[c] = float(len(y[y == c])) / len(y) # P(y=label)


    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y)



if __name__ == '__main__':
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train/255, X_test/255
    X_train = np.reshape(X_train, newshape=(-1, 28**2))
    X_test = np.reshape(X_test, newshape=(-1, 28**2))

    nb_model = NaiveBayes()
    nb_model.fit(X_train, y_train)
    b_model = Bayes()
    b_model.fit(X_train, y_train)
    print(f'Naive-Bayes Train score: {round(nb_model.score(X_train, y_train), 2)}')
    print(f'Naive-Bayes Test score: {round(nb_model.score(X_test, y_test), 2)}')
    print(f'Bayes Train score: {round(b_model.score(X_train, y_train), 2)}')
    print(f'Bayes Test score: {round(b_model.score(X_test, y_test), 2)}')