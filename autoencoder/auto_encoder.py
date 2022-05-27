import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import colors

#Params
nr_epochs = 500
lr = 0.2

#Reading toy data
df = pd.read_csv('toy_data.csv')
ns = 1500
dat = np.zeros((ns,100))
for i in range(0, ns):
    dat[i, :] = df['2'].values + 2.0 * np.random.randn(len(df['2']))

inputs = tf.cast(dat, tf.float32)

# Set up NN
model_tf = tf.keras.Sequential()
# model_tf.add(tf.keras.layers.Dense(10, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(1.5)))
model_tf.add(tf.keras.layers.Dense(100, activation='sigmoid'))
opt = tf.optimizers.Adam(learning_rate=lr)
model_tf.compile(optimizer = opt, loss='MSE')
model_tf.fit(inputs, inputs, epochs=nr_epochs, verbose=1)


fig, axs = plt.subplots(2, 2)
x = np.reshape(df['2'].values,(10,10))
axs[0, 0].imshow(x, cmap=colors.ListedColormap(['red', 'white']))
ys = model_tf.predict(tf.cast(np.expand_dims(df['2'].values, 0), tf.float32))
xh = np.reshape(ys,(10,10))
axs[1, 0].imshow(xh, cmap=colors.ListedColormap(['red', 'white']))

x = np.reshape(df['3'].values,(10,10))
axs[0, 1].imshow(x, cmap=colors.ListedColormap(['blue', 'white']))
ys = model_tf.predict(tf.cast(np.expand_dims(df['3'].values, 0), tf.float32))
xh = np.reshape(ys,(10,10))
axs[1, 1].imshow(xh, cmap=colors.ListedColormap(['blue', 'white']))
plt.show()
