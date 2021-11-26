from KNN import KNN
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

X = np.zeros((200, 2))
X[:50] = np.random.random((50, 2))/2 + 0.5 #
X[50:100] = np.random.random((50, 2))/2 + 0.5 #
X[100:150] = np.random.random((50, 2))/2 + np.array([[0, 0.5]])
X[150:] = np.random.random((50, 2))/2 + np.array([[0.5, 0]])
Y = np.reshape(np.array([0] * 100 + [1] * 100), newshape=(-1, 1))

def accuracy_score(y_true, y_pred):
    return y_pred[y_pred[:, 0] == y_true[:, 0]].shape[0] / y_true.shape[0]

plt.scatter(x=X[:, 0], y=X[:, 1], s=100, c=Y)
plt.show()

knn = KNN(k=3)
knn.fit(X, Y)
y_pred = knn.predict(X)
print(f'Accuracy Score: {accuracy_score(y_true=Y, y_pred=y_pred)}')

