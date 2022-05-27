import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns=None, label_columns=None):

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.train_label_indices = {name: i for i, name in enumerate(train_df.columns)}

    # ...and the input column indices
    self.input_columns = input_columns
    if input_columns is not None:
      self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
    self.train_input_indices = {name: i for i, name in enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.input_columns is not None:
        inputs = tf.stack([inputs[:, :, self.train_input_indices[name]] for name in self.input_columns], axis=-1)
      if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.train_label_indices[name]] for name in self.label_columns], axis=-1)
      inputs_flat = tf.reshape(inputs, shape=[tf.shape(inputs)[0], -1])
      labels_flat = tf.reshape(labels, shape=[tf.shape(labels)[0], -1])
      return inputs_flat, labels_flat


  def make_dataset(self, data, shuffle = False, batchsize = 500,):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                                sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize)
      ds = ds.map(self.split_window)
      return ds


if __name__ == '__main__':
    # artificial data
    per = 50  # period length
    dmp = 0.95  # damping
    a21 = -dmp ** 2
    a22 = 2 * dmp * np.cos(2 * np.pi / per)
    A = np.array([[0, 1], [a21, a22]])
    b = np.array([0, 1])
    c = np.array([1, 0])
    g = np.matmul(c, np.matmul(np.linalg.inv(np.eye(2) - A), b))  # stationary gain input -> output

    # state space model
    sim_steps = 150
    x = np.zeros([sim_steps, 2])    # state
    x[0, :] = [0, 0]                # initial state
    u = np.random.standard_normal(sim_steps)  # input (white noise)
    y = np.zeros([sim_steps, 1])    # output (colored noise)
    y2 = np.zeros([sim_steps, 1])
    y3 = np.zeros([sim_steps, 1])
    for i in range(1, sim_steps):
        x[i, :] = np.matmul(A, x[i - 1, :]) + b / g * u[i - 1]
        y[i, 0] = np.matmul(c, x[i, :])
        if i>1:  # equivalent ARMA representation
            y2[i, 0] = a21 * y2[i-2, 0] + a22 * y2[i-1, 0] + 1 / g * u[i-2]
        if i>3:  # (truncated) impulse response representation
            y3[i, 0] = 1 / g * u[i-2] + a22 / g * u[i-3] + (a21 + a22**2) / g * u[i-4]

    # plotting
    [v, c] = np.linalg.eig(A)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01)))
    axs[0].scatter([x.real for x in v], [x.imag for x in v])
    axs[0].set_aspect('equal', 'box')
    axs[0].set_title('spectrum')
    axs[1].plot(range(0, sim_steps), y)
    axs[1].plot(range(0, sim_steps), y2,'g--')
    #axs[1].plot(range(0, sim_steps), y3,'r-',)
    axs[1].set_title('simulation')
    plt.show()

    # prepare data
    dat = np.concatenate((np.expand_dims(u, 1), y2), axis=1)
    df = pd.DataFrame(dat, columns = ['u', 'y'])
    n = len(df)
    train_df = df[0:int(n*0.8)]
    val_df = df[int(n*0.8):]

    # sliding window
    lb = 2
    window = WindowGenerator(input_width = lb, label_width = 1, shift=1, label_columns=['y'])
    train_data = window.make_dataset(train_df)
    val_data = window.make_dataset(val_df)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation=None))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=0.01), metrics=[tf.metrics.MeanSquaredError()])
    model.run_eagerly = True
    history = model.fit(train_data, epochs=350, validation_data=train_data, callbacks=[early_stopping])

    plt.figure()
    plt.plot(history.history['loss'])

    plt.figure()
    y_pred = model.predict(train_data)

    u_true = np.concatenate([x for x, y in train_data], axis=0)
    y_true = np.concatenate([y for x, y in train_data], axis=0)
    plt.plot(y_true[:, -1])
    plt.plot(y_pred[:, -1],'--')
    plt.legend(['y_pred', 'y_true'])
    plt.show()

    params_estimated = model.layers[0].trainable_variables[0].numpy()
    params_true = [[1 / g, a21, 0, a22]]
    print('estimated =\n', np.around(np.transpose(params_estimated), 3))
    print('ground truth =\n', np.around(params_true, 3))
