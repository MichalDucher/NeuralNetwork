import numpy as np
import pandas as pd

train_data = np.array(pd.read_csv('mnist_train.csv')).T
test_data = np.array(pd.read_csv('mnist_test.csv')).T

X_train, y_train = train_data[1:] / 255, train_data[0]
X_test, y_test = test_data[1:] / 255, test_data[0]

from NN import Network

network = Network()
network.fit(X_train, y_train)
predicted = network.predict(X_test)
acc = network.get_accuracy(predicted, y_test)
print("Acc: ", acc)