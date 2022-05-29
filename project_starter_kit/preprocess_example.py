import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns=None, label_columns=None, all_columns=None):

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.train_label_indices = {name: i for i, name in enumerate(all_columns)}

    # ...and the input column indices
    self.input_columns = input_columns
    if input_columns is not None:
      self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
    self.train_input_indices = {name: i for i, name in enumerate(all_columns)}

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
      #inputs.set_shape([None, self.input_width, None])
      #labels.set_shape([None, self.label_width, None])
      return inputs, labels

  def make_dataset(self, data, shuffle = False, batchsize = 500,):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                                sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize)
      ds = ds.map(self.split_window)
      return ds


class Autoencoder(tf.keras.models.Model):
  def __init__(self, num_timesteps, num_inputs, num_hidden, kernel_size, pooling):
    super(Autoencoder, self).__init__()
    self.num = num_timesteps
    self.lb = kernel_size
    self.pooling =pooling

    encoder_input = tf.keras.Input(shape=(num_timesteps, num_inputs), name="input")
    x = tf.keras.layers.Conv1D(filters=num_hidden, kernel_size=kernel_size, activation=None, use_bias=True, padding='causal')(encoder_input)
    x = layers.MaxPooling1D(self.pooling, strides=self.pooling, padding='same')(x)
    self.encoder = tf.keras.Model(inputs=encoder_input, outputs=x)
    decoder_input = tf.keras.Input(shape=(int(num_timesteps/self.pooling), num_hidden))
    y = tf.keras.layers.Conv1DTranspose(filters=num_inputs, kernel_size=kernel_size, strides=self.pooling, activation=None, use_bias=True, padding='same')(decoder_input)
    self.decoder = tf.keras.Model(inputs=decoder_input, outputs=y)

  def call(self, input):
    u = self.encoder(input)
    decoded = self.decoder(u)
    return decoded

if __name__ == '__main__':
    # prepare data
    df_ = pd.read_excel('./market_data_update.xlsx',sheet_name='market_data', engine = 'openpyxl')
    df_ = df_.set_index(df_['Date'])
    df_ = df_.drop(columns='Date')
    dat = df_[['VOL', '_SPX']]
    dat = dat.rename(columns={'VOL': 'u', '_SPX': 'y'})
    df = (dat - dat.mean()) / dat.std()

    # prepare de-noising set-up
    df_n = df + 1.0 * np.random.normal(0, 1, df.shape)
    df[['u_n','y_n']] = df_n[['u','y']].copy()
    n = len(df)

    # sliding window
    lb = 30
    pooling = 1
    window = WindowGenerator(input_width=lb, label_width=lb, shift=0, input_columns=['u_n','y_n'], label_columns=['u','y'], all_columns=df.columns)
    td = window.make_dataset(df, shuffle=True)
    train_data = td.take(2)
    val_data = td.skip(2)

    # Training; Hint: play with num_hidden = 1 or 2, and kernel_size
    model = Autoencoder(num_timesteps=lb, num_inputs=2, num_hidden=2, kernel_size=25, pooling=pooling)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=50, decay_rate=0.97, staircase=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError(), ])
    model.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, mode='min')
    history = model.fit(train_data, validation_data=val_data, epochs=150, callbacks=[early_stopping])
    model.summary()

    fig, axs = plt.subplots()
    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.legend(['training loss', 'validation loss'])

    # PLOTTING RECONSTRUCTION RESULTS and SAVE as preprocessed data
    window_orig = WindowGenerator(input_width=lb, label_width=lb, shift=0, input_columns=['u', 'y'],label_columns=['u', 'y'], all_columns=df.columns)
    fd = window_orig.make_dataset(df, shuffle=False)

    y_pred = model.predict(fd)
    u_true = np.concatenate([x for x, y in fd], axis=0)
    y_true = np.concatenate([y for x, y in fd], axis=0)
    mse_FD = ((y_pred-y_true)**2).mean()
    print('mse_FD ' + str(mse_FD))

    plt.figure()
    plt.subplot(3,1,1)
    mp = -1
    du = pd.DataFrame(y_true[:, mp, :], index=df.index[lb - 1:])
    dy = pd.DataFrame(y_pred[:, mp, :], index=df.index[lb - 1:])

    plt.plot(du.iloc[:,0],'-', linewidth = 1)
    plt.plot(dy.iloc[:,0],'-', linewidth = 1)
    plt.legend(['y1_true', 'y1_pred'])

    plt.subplot(3,1,2)
    plt.plot(du.iloc[:,1],'-', linewidth = 1)
    plt.plot(dy.iloc[:,1],'-', linewidth = 1)
    plt.legend(['y2_true', 'y2_pred'])

    middle = model.encoder(u_true)
    plt.subplot(3,1,3)
    plt.plot((pd.DataFrame(middle[:,mp,:], index=df.index[lb - 1:])))
    plt.legend(['middle layer'])
    plt.show()

    # Note: output file should contain smoothed and original data
    pd.merge(dy, dat.iloc[lb - 1:, :], on="Date").to_excel('data_preprocessed.xlsx')


