import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

#Params
nr_epochs = 5000
lr = .1

#Read the data
df = pd.read_excel('toy_data.xls')

# Assign in and output
inputs = df[['x1', 'x2']]
inputs = tf.cast(inputs, tf.float32)
labels = df['y'] > 0

# Set up NN
model_tf = tf.keras.Sequential()
model_tf.add(tf.keras.layers.Dense(3, activation='sigmoid'))
model_tf.add(tf.keras.layers.Dense(2, activation='sigmoid'))
model_tf.add(tf.keras.layers.Dense(1, activation='sigmoid'))
opt = tf.optimizers.Adam(learning_rate=lr)
model_tf.compile(optimizer = opt,loss='binary_crossentropy')
model_tf.fit(inputs, labels, epochs=nr_epochs, verbose=1)

plt.figure()
plt.plot(labels)
# eval model
pred_eval = model_tf.predict(inputs)
acc = accuracy_score(labels, np.round(pred_eval))
plt.plot(np.round(pred_eval))
plt.legend(['ground truth', 'model' ])
plt.title('accuracy NN model  : ' + str(np.round(acc,2)))


plt.figure()
plt.scatter(inputs[:49, 0], inputs[:49, 1], c='b')
plt.scatter(inputs[49:, 0], inputs[49:, 1], c='r')

# test model
[x0, x1] = np.meshgrid(np.arange(0,1,0.05),np.arange(0,1,0.05))
test_data = tf.cast(np.transpose(np.array([x0, x1]).reshape(2, x0.shape[0] ** 2)), tf.float32)
pred_test = model_tf.predict(test_data)
x2 = pred_test.reshape(x0.shape[0], x0.shape[0])
plt.contour(x0, x1, x2)
plt.show()


