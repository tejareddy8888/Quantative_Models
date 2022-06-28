import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import tensorflow as tf


def load_data():
    # prepare data
    data_df = pd.read_excel('./market_data.xlsx',
                            sheet_name='data', engine='openpyxl')
    data_df.set_index('Date', drop=True, inplace=True)
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

        self.all_column_indices = {
            name: i for i, name in enumerate(all_columns)}

        # input parameters to configure the window.
        self.input_width = input_width
        self.label_width = label_width
        # offset or shift before the prediction label value.
        self.shift = shift
        self.total_window_size = input_width + shift

        # the actual length on input size to be taken on every window on the data
        self.input_slice = slice(0, input_width)
        # derive the input indices on the data generated after the window application
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        # derive the label starting postion
        self.label_start = self.total_window_size - self.label_width
        # the actual length on label size to be taken on every window on the data
        self.labels_slice = slice(self.label_start, None)
        # derive the label indices on the data generated after the window application
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

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
            inputs = tf.stack([inputs[:, :, self.all_column_indices[name]]
                              for name in self.input_columns], axis=-1)
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.all_column_indices[name]]
                              for name in self.label_columns], axis=-1)
        #inputs.set_shape([None, self.input_width, None])
        #labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data, shuffle=False, batchsize=500,):
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
        self.pooling = pooling

        encoder_input = tf.keras.Input(
            shape=(num_timesteps, num_inputs), name="input")
        x = tf.keras.layers.Conv1D(filters=num_hidden, kernel_size=kernel_size,
                                   activation=None, use_bias=True, padding='causal')(encoder_input)
        x = tf.keras.layers.MaxPooling1D(
            self.pooling, strides=self.pooling, padding='same')(x)
        self.encoder = tf.keras.Model(inputs=encoder_input, outputs=x)
        decoder_input = tf.keras.Input(
            shape=(int(num_timesteps/self.pooling), num_hidden))
        y = tf.keras.layers.Conv1DTranspose(filters=num_inputs, kernel_size=kernel_size,
                                            strides=self.pooling, activation=None, use_bias=True, padding='same')(decoder_input)
        self.decoder = tf.keras.Model(inputs=decoder_input, outputs=y)

    def call(self, input):
        u = self.encoder(input)
        decoded = self.decoder(u)
        return decoded


if __name__ == '__main__':
    through_cnn = True
    timesteps = 5
    pooling = 1

    input_columns = ['Rho', 'CPI', '_MKT']
    target_columns = ['_MKT']

    # Load the data from the sheet
    data = load_data()
    # data['VIX_phase'] = data.apply(
    #     lambda x: 'low' if x['VOL'] < 12 else 'high' if x['VOL'] > 20 else 'medium', axis=1)

    # Categorize phase variables into thresholds. for M2 and OIL, separate into mean +- 0.5 standard deviation
    data['VIX_phase'] = data.apply(
        lambda x: 'low' if x['VOL'] < 12 else 'high' if x['VOL'] > 20 else 'medium', axis=1
    )
    data['M2_phase'] = data.apply(
        lambda x: 'low' if x['M2'] < (data['M2'].mean()-(0.5*data['M2'].std())) else 'high' if x['M2'] > (data['M2'].mean()+(0.5*data['M2'].std())) else 'medium', axis=1 
    )
    data['_OIL_phase'] = data.apply(
        lambda x: 'low' if x['_OIL'] < (data['_OIL'].mean()-(0.5*data['_OIL'].std())) else 'high' if x['_OIL'] > (data['_OIL'].mean()+(0.5*data['_OIL'].std())) else 'medium', axis=1 
    )
    print(data.VIX_phase.value_counts())
    # Normalize the entire dataset,
    data = normalize_data(data)

    # creating a sliding windowed data
    # Few constants like input window of number of time steps width and prediction timesteps width
    # Width = timesteps +1 because need to include the target at T
    window = WindowGenerator(input_width=(timesteps+1), label_width=1, shift=1,
                             input_columns=input_columns, label_columns=target_columns, all_columns=data.columns)

    # Below is to check the window generation if needed.
    # for batch in windowed_data:
    #   inputs, labels = batch
    #   print("Inputs :",inputs[-1])
    #   print("Lables : ",labels[-1])
    # sliding window

    td = window.make_dataset(data, True, 250)
    index = 0
    for i in td:
        index += 1
        print(index)
    train_data = td.take(4)
    val_data = td.skip(4)

    checkpoint_path = "checkpoint/autoencoder.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

    train = tf.concat([x for x, _ in train_data], axis=0)

    val = tf.concat([x for x, _ in val_data], axis=0)

    # # Training; Hint: play with num_hidden = 1 or 2, and kernel_size
    # time steps +1 because of the same reason as when defining window
    AC = Autoencoder(num_timesteps=(timesteps+1), num_inputs=len(
        input_columns), num_hidden=2, kernel_size=25, pooling=pooling)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.01, decay_steps=50, decay_rate=0.97, staircase=True)
    AC.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(
        learning_rate=lr_schedule))
    AC.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=50, mode='min')
    AC_history = AC.fit(x=train, y=train, validation_data=(val, val), epochs=100, callbacks=[
                        early_stopping, checkpoint_callback])
    AC.summary()

    # # # Loads the weights
    # AC.load_weights(checkpoint_path)

    # Plot loss from Auto Encoder
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(AC_history.history['loss'])
    axs[0].plot(AC_history.history['val_loss'])
    axs[0].legend(['training loss', 'validation loss'])

    # Plot the autoencoded data and actual data
    # flattened_data = tf.stack(tf.convert_to_tensor(train_data.as_numpy_iterator()))

    # below is to compare the autoencoded_train_data against actual train data

    autoencoded_train_inputs = AC.predict(train)
    autoencoded_val_inputs = AC.predict(val)
    # dumpy_tensor

    axs[2].plot(val[:,-1,-1])
    axs[2].plot(autoencoded_val_inputs[:,-1,-1])
    axs[2].legend(['training signal', 'autoencoded signal'])
    
    # Separate x_train, x_val and y_train from autoecoded data
    x_train= tf.convert_to_tensor([x[:5] for x in autoencoded_train_inputs])
    x_val= tf.convert_to_tensor([x[:5] for x in autoencoded_val_inputs])
    y_train = tf.convert_to_tensor([y[5][2] for y in autoencoded_train_inputs])
    # y_val is run throught auto encoder
    y_val = tf.convert_to_tensor([[[y[5][2]]] for y in val])

    # define sliding window
    lf = 1      # look forward
    ks = 3      # kernel size
    lw = 1      # label width
    lb = timesteps
    # Train with RNN
    if through_cnn:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(
            filters=4, kernel_size=ks, activation='relu', use_bias=False))
        model.add(tf.keras.layers.MaxPooling1D(
            pool_size=4, strides=1, padding='same'))
        model.add(tf.keras.layers.Conv1D(
            filters=32, kernel_size=ks, activation='relu', use_bias=False))
        model.add(tf.keras.layers.MaxPooling1D(
            pool_size=4, strides=1, padding='same'))
        model.add(tf.keras.layers.Conv1D(filters=4, kernel_size=lb -
                  2*(ks-1)-(lw-1), activation='relu', use_bias=False))
        model.add(tf.keras.layers.Dense(units=1))

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.01, decay_steps=150, decay_rate=0.95, staircase=True)
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(
            learning_rate=lr_schedule))
        model.run_eagerly = False
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=100, mode='min')
        model_history = model.fit(
            x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=750, batch_size=150, callbacks=[early_stopping])
        print(model.summary())

    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(3, return_sequences=False, return_state=False, activation=None, use_bias=False))
        model.add(tf.keras.layers.Dense(10, activation=None, use_bias=False))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(1, activation=None, use_bias=False))
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=0.001))
        model.run_eagerly = False
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')
        model_history = model.fit(x=x_train, y=y_train, validation_data=(x_val,y_val), epochs=500, callbacks=[early_stopping])
        print(model.summary())

        # Plot loss from RNN
    axs[1].plot(model_history.history['loss'])
    axs[1].plot(model_history.history['val_loss'])
    axs[1].legend(['training loss', 'validation loss'])

    plt.show()
