import numpy as np
import pandas as pd


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), f'X must be a Numpy array.'
        assert isinstance(y, np.ndarray), f'y must be a Numpy array.'
        assert X.ndim == 2, f'The array must have 2 dimensions.'
        assert y.ndim == 2, f'The array must have 2 dimensions.'
        self.X = X
        self.y = y

    def predict(self, X):
        y = []
        for x in X:
            d = np.sqrt(np.sum(np.power((x - self.X), 2), axis=1))
            d = np.concatenate((np.reshape(d, newshape=(-1, 1)), self.y), axis=1)
            d = d[d[:, 0].argsort()][:self.k]
            y.append(np.bincount(d[:, 1].astype(int)).argmax())
        return np.reshape(np.array(y), newshape=(-1, 1))


if __name__ == '__main__':
    cols = ['SEPAL_LENGTH', 'SEPAL_WIDTH', 'PETAL_LENGTH', 'PETAL_WIDTH', 'CLASS']
    df = pd.read_csv(r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv')
    df.columns = cols
    labels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    df['CLASS'] = df['CLASS'].map(labels)
    df = df.sample(frac=1, random_state=1234) # Shuffling the data

    def train_test_split(X, y, train_size=0.8):
        size = int(X.shape[0]*train_size)
        return X[:size], X[size:], y[:size], y[size:]


    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='CLASS').values, df.CLASS.values)
    y_train, y_test = np.reshape(y_train, newshape=(-1, 1)), np.reshape(y_test, newshape=(-1, 1))

    def accuracy_score(y_true, y_pred):
        return y_pred[y_pred[:, 0] == y_true[:, 0]].shape[0] / y_true.shape[0]


    for k in [3, 4, 5, 6, 7, 8, 9, 10]:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(f'k = {k} - Accuracy: {round(accuracy_score(y_test, y_pred), 4)}')