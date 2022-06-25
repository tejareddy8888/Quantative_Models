import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import tensorflow as tf


def load_data():
    # prepare data
    data_df = pd.read_excel('./market_data.xlsx',sheet_name='data', engine = 'openpyxl')
    data_df.set_index('Date',drop=True,inplace=True)
    return data_df

def normalize_data(df):
    """ This function can also be called as feature scaling like normalization, min-max scaling, 
    also test sklearn.preprocessing import StandardScaler or other preprocessing
    """
    df = (df - df.mean()) / df.std()
    return df


class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns, label_columns, all_columns):

    # Derive the label/target column indices.
    self.label_columns = label_columns

    # Derive feature/signal/input column indices
    self.input_columns = input_columns

    self.all_column_indices = {name: i for i, name in enumerate(all_columns)};


    # input parameters to configure the window.
    self.input_width = input_width
    self.label_width = label_width
    ## offset or shift before the prediction label value.
    self.shift = shift
    self.total_window_size = input_width + shift

    # the actual length on input size to be taken on every window on the data
    self.input_slice = slice(0, input_width)
    # derive the input indices on the data generated after the window application
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    # derive the label starting postion
    self.label_start = self.total_window_size - self.label_width
    # the actual length on label size to be taken on every window on the data
    self.labels_slice = slice(self.label_start, None)
    # derive the label indices on the data generated after the window application
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Input columns: {self.input_columns}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.input_columns is not None:
        inputs = tf.stack([inputs[:, :, self.all_column_indices[name]] for name in self.input_columns], axis=-1)
      if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.all_column_indices[name]] for name in self.label_columns], axis=-1)
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
    x = tf.keras.layers.MaxPooling1D(self.pooling, strides=self.pooling, padding='same')(x)
    self.encoder = tf.keras.Model(inputs=encoder_input, outputs=x)
    decoder_input = tf.keras.Input(shape=(int(num_timesteps/self.pooling), num_hidden))
    y = tf.keras.layers.Conv1DTranspose(filters=num_inputs, kernel_size=kernel_size, strides=self.pooling, activation=None, use_bias=True, padding='same')(decoder_input)
    self.decoder = tf.keras.Model(inputs=decoder_input, outputs=y)

  def call(self, input):
    u = self.encoder(input)
    decoded = self.decoder(u)
    return decoded


if __name__ == '__main__':
    timesteps = 5
    pooling = 1

    input_columns = ['Rho', 'CPI','_MKT']
    target_columns = ['_MKT']

    ## Load the data from the sheet
    data = load_data()

    ## Normalize the entire dataset,
    data = normalize_data(data)

    ## creating a sliding windowed data
    # Few constants like input window of number of time steps width and prediction timesteps width
    window = WindowGenerator(input_width=timesteps, label_width=1, shift=1, input_columns=input_columns, label_columns=target_columns, all_columns=data.columns)

    # Below is to check the window generation if needed.
    # for batch in windowed_data:
    #   inputs, labels = batch
    #   print("Inputs :",inputs[-1])
    #   print("Lables : ",labels[-1])
        # sliding window

    td = window.make_dataset(data)
    train_data = td.take(2)
    val_data = td.skip(2)

    checkpoint_path = "checkpoint/autoencoder.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Training; Hint: play with num_hidden = 1 or 2, and kernel_size
    model = Autoencoder(num_timesteps=timesteps, num_inputs=len(input_columns), num_hidden=2, kernel_size=25, pooling=pooling)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=50, decay_rate=0.97, staircase=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError(), ])
    model.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, mode='min')
    history = model.fit(train_data, validation_data=val_data, epochs=500, callbacks=[early_stopping, checkpoint_callback])
    model.summary()


    # Loads the weights
    model.load_weights(checkpoint_path)

    

    fig, axs = plt.subplots()
    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.legend(['training loss', 'validation loss'])
    
    # fig, axs = plt.subplots()
    # data.loc[:,['DY',]].plot(ax=axs)
    # plt.show()



## Split the data into train test and validation using the windowgenerator as we want the time series methodology while doing this also refer the documents
## perform pre-classification using the label encoder or any other logic like logistic regression or any classification logic on three signals CPI, Rho, DIL
## perform Dimensional Reduction 
#     compare the fitting vs overfitting
# denoising using the autoencoder
# kalman filter
# seperating the signal from noise


# irrespective of the activation function, applying pooling on the 

# drop out layers help in creating subgraph and model averaging kind of concepts
# information coefficient
# bayesian inferance
# prior psoterior
# granger causality


# Data Ingestion
#     - Data Splitting for validation and
# Data Preprocessing
#     - Denoising AutoEncoder
#     - Kalman Filter
#     - Dimensional Reduction

# Model: 
#     - Training Strategies
#     - Cross Validation
#     - Trading strategies
#           - Supply shock
#           - Causality, find the casual dimension
#           - 
#     - LOSS Functions
#     - Optimizer choice
#     - Learning Rate scheduler

# Plotting Data in everyphase

# clustering the 