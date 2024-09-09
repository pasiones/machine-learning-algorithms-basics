import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict_1 = unpickle('C:/Users/Admin/OneDrive/Documents/machine-learning-algorithms-basics/week05/cifar-10-python/cifar-10-batches-py/data_batch_1')
datadict_2 = unpickle('C:/Users/Admin/OneDrive/Documents/machine-learning-algorithms-basics/week05/cifar-10-python/cifar-10-batches-py/data_batch_2')
datadict_3 = unpickle('C:/Users/Admin/OneDrive/Documents/machine-learning-algorithms-basics/week05/cifar-10-python/cifar-10-batches-py/data_batch_3')
datadict_4 = unpickle('C:/Users/Admin/OneDrive/Documents/machine-learning-algorithms-basics/week05/cifar-10-python/cifar-10-batches-py/data_batch_4')
datadict_5 = unpickle('C:/Users/Admin/OneDrive/Documents/machine-learning-algorithms-basics/week05/cifar-10-python/cifar-10-batches-py/data_batch_5')
datadict_test = unpickle('C:/Users/Admin/OneDrive/Documents/machine-learning-algorithms-basics/week05/cifar-10-python/cifar-10-batches-py/test_batch')

X_1 = datadict_1["data"]
Y_1 = datadict_1["labels"]
X_2 = datadict_2["data"]
Y_2 = datadict_2["labels"]
X_3 = datadict_3["data"]
Y_3 = datadict_3["labels"]
X_4 = datadict_4["data"]
Y_4 = datadict_4["labels"]
X_5 = datadict_5["data"]
Y_5 = datadict_5["labels"]

X = np.concatenate((X_1, X_2, X_3, X_4, X_5))
Y = np.concatenate((Y_1, Y_2, Y_3, Y_4, Y_5))

X_test = datadict_test["data"]
Y_test = datadict_test["labels"]

X_tr_normalized = X / 255
Y_tr_one_hot = np.zeros((Y.shape[0], 10))
for i in range(Y.shape[0]):
    class_id = Y[i]
    Y_tr_one_hot[i][class_id] = 1

X_test_normalized = X_test / 255

model = Sequential()

model.add(Dense(100, input_dim = 3072, activation = 'sigmoid'))
model.add(Dense(500, activation = 'sigmoid'))
model.add(Dense(10, activation = 'softmax'))

lr = 0.05
epochs = 50
opt = keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer=opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

tr_hist = model.fit(X_tr_normalized, Y_tr_one_hot, epochs = epochs, verbose = 1)

Y_tr_predict = model.predict(X_tr_normalized)
Y_tr_predict_classes = [np.argmax(element) for element in Y_tr_predict]
correct = accuracy_score(Y, Y_tr_predict_classes)
print(f"Classification accuracy (training data): {correct*100:.2f}%")

Y_te_predict = model.predict(X_test_normalized)
Y_te_predict_classes = [np.argmax(element) for element in Y_te_predict]
correct = accuracy_score(Y_test, Y_te_predict_classes)
print(f"Classification accuracy (test data): {correct*100:.2f}%")

plt.plot(tr_hist.history['loss'])
plt.title(f'Learning rate = {lr} and epochs = {epochs}')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# Case 1: learning rate = 0.01, epoch = 10
# Accuracy for training data = 32.60%, test data = 32.69%

# Case 2: learning rate = 0.01, epoch = 50
# Accuracy for training data = 48.19%, test data = 46.09%

# Case 3: learning rate = 0.05, epoch = 10
# Accuracy for training data = 41.68%, test data = 41.37%

# Case 4: learning rate = 0.05, epoch = 50
# Accuracy for training data = 59.14%, test data = 49.95%