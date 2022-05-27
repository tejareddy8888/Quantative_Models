import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

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

      return inputs, labels

  def make_dataset(self, data, shuffle = False, batchsize = 500,):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                                sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize)
      ds = ds.map(self.split_window)
      return ds


class Autoencoder(tf.keras.models.Model):
  def __init__(self, num_timesteps, num_inputs, num_hidden, kernel_size, pooling, num_outputs):
    super(Autoencoder, self).__init__()
    self.num = num_timesteps
    self.lb = kernel_size
    self.pooling =pooling

    encoder_input = tf.keras.Input(shape=(num_timesteps, num_inputs), name="input")
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, activation=None, use_bias=False, padding='same')(encoder_input)
    x = layers.MaxPooling1D(self.pooling, padding="same")(x)
    self.encoder = tf.keras.Model(inputs=encoder_input, outputs=x)

    decoder_input = tf.keras.Input(shape=(int(num_timesteps/self.pooling), num_hidden))
    y = tf.keras.layers.Conv1DTranspose(filters=2, kernel_size=kernel_size, strides = self.pooling, activation=None, use_bias=True, padding='same')(decoder_input)
    self.decoder = tf.keras.Model(inputs=decoder_input, outputs=y)

  def call(self, input):
    u = self.encoder(input)
    decoded = self.decoder(u)
    return decoded

if __name__ == '__main__':
    # prepare data
    df_ = pd.read_excel('market_data.xlsx', engine = 'openpyxl')
    df_ = df_.set_index(df_['Date'])
    df_ = df_.drop(columns='Date')
    dat = df_[['VOL', '_MKT']]
    dat = dat.rename(columns={'VOL': 'u', '_MKT': 'y'})
    df = (dat - dat.mean()) / dat.std()

    df_n = df + np.random.normal(0, 1, df.shape)
    df['u_n'] = df_n['u'].copy()
    df['y_n'] = df_n['y'].copy()

    n = len(df)
    train_df = df[0:int(n*1)]
    val_df = df[int(n*0.8):]
    train_dfn = df[0:int(n*1)]

    # sliding window
    lb = 100
    pooling = 2
    window = WindowGenerator(input_width=lb, label_width=lb, shift=0, input_columns=['u_n','y_n'], label_columns=['u','y'])
    train_data = window.make_dataset(train_df)
    val_data = window.make_dataset(val_df)
    # model = Autoencoder(num_timesteps=lb, num_inputs=2, num_hidden=1, kernel_size=10, num_outputs= 1)
    model = Autoencoder(num_timesteps=lb, num_inputs=2, num_hidden=1, kernel_size=10, num_outputs= 1, pooling=pooling)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=50, decay_rate=0.97, staircase=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError(), ])
    model.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, mode='min')
    history = model.fit(train_data, epochs=100, callbacks=[early_stopping])
    model.summary()

    fig, axs = plt.subplots()
    axs.plot(history.history['loss'])

    window2 = WindowGenerator(input_width=lb, label_width=lb, shift=0, input_columns=['u', 'y'], label_columns=['u', 'y'])
    train_data_no_noise = window2.make_dataset(train_df)

    y_pred = model.predict(train_data)
    u_true = np.concatenate([x for x, y in train_data], axis=0)
    y_true = np.concatenate([y for x, y in train_data], axis=0)
    u_true_no_noise = np.concatenate([x for x, y in train_data_no_noise], axis=0)

    plt.figure()
    plt.subplot(3,1,1)
    mp = int(lb/pooling/2) -1
    plt.plot((u_true[:, mp, 0]),'--', linewidth = .5)
    plt.plot((u_true_no_noise[:, mp, 0]), linewidth = 1)
    plt.plot((y_pred[:, mp, 0]), linewidth = 2)
    plt.legend(['u_true with noise', 'u_true', 'u_pred'])

    plt.subplot(3,1,2)
    plt.plot(np.cumsum(u_true[:, mp, -1]),'--',linewidth = .5)
    plt.plot(np.cumsum(u_true_no_noise[:, mp, -1]),linewidth = 1)
    plt.plot(np.cumsum(y_pred[:, mp, -1]), linewidth = 2)
    plt.legend(['y_true with noise', 'y_true', 'y_pred'])

    middle = model.encoder(u_true)
    plt.subplot(3,1,3)
    plt.plot((pd.DataFrame(middle[:,mp,-1])))
    plt.legend(['middle layer'])

    pd.DataFrame(y_pred[:, mp, :], index = df.index[lb-1:]).to_excel('data_denoised.xlsx')
    plt.show()

