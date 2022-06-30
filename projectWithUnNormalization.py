import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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

        ## input (5,3)
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


def load_data(testSize):
    data_df = pd.read_excel('./market_data.xlsx', sheet_name='data', engine='openpyxl')
    data_df.set_index('Date', drop=True, inplace=True)
    return data_df[:-testSize], data_df[-testSize:]


def convert_and_add_phases(array, columns, index):
    df = pd.DataFrame(array,columns=columns,index=index)
    # Create new column for each phase variable. 3 threshold, high, normal, low.
    df['VIX_phase'] = df.apply(
        lambda x: 'low' if x['VOL'] < 12 else 'high' if x['VOL'] > 20 else 'medium', axis=1
    )
    df['M2_phase'] =df.apply(
        lambda x: 'low' if x['M2'] < (df['M2'].mean()-(0.5*df['M2'].std())) else 'high' if x['M2'] > (df['M2'].mean()+(0.5*df['M2'].std())) else 'medium', axis=1
    )
    df['_OIL_phase'] = df.apply(
        lambda x: 'low' if x['_OIL'] < (df['_OIL'].mean()-(0.5*df['_OIL'].std())) else 'high' if x['_OIL'] > (df['_OIL'].mean()+(0.5*df['_OIL'].std())) else 'medium', axis=1
    )
    return df

def graph_phase_variables(phase_variables):
    df = pd.concat([train_df, test_df],axis=0)
    # Graph of phase variables with its thresholds
    for variable in phase_variables:
        plt.figure()
        plt.subplot()
        plt.plot(df[variable])
        plt.axhline(y=(df[variable].mean()+(0.5*df[variable].std())), color='g', linestyle='-')
        plt.axhline(y=(df[variable].mean()-(0.5*df[variable].std())), color='r', linestyle='-')
        plt.legend([variable,'upperbound','lowerbound'])
        plt.show()

def graph_normalized_and_autoencoded():
    # Graph of forecasting variables and how autoencoder smooth them
    df = pd.concat([train_df, test_df],axis=0)
    variables = window.make_dataset(np.concatenate([train_array,test_array],axis=0), shuffle=False)
    autoencoded_variables = AC.predict(variables)
    for var_name in input_columns:
        variable = np.array((df[var_name]- df[var_name].mean())/df[var_name].std())
        index = input_columns.index(var_name)
        autoencoded = autoencoded_variables[:,timesteps,index]
        plt.figure()
        plt.subplot()
        plt.plot(variable[timesteps:])
        plt.plot(autoencoded)
        plt.legend([" "+var_name,'Autoencoded '+var_name])
        plt.show()


if __name__ == '__main__':
    testSize = 250
    timesteps = 7
    pooling = 1
    input_columns = ['MOV ', 'VOL', 'Rho', 'CPI', '_MKT',]
    target_columns = ['_MKT']
    phase_variables = ['M2','_OIL','VOL']

    # Load the data from the sheet
    train_df, test_df = load_data(testSize)
    # Graph phase variables and its thresh holds
    graph_phase_variables(phase_variables)
    # Normalize the entire dataset,
    scaler = StandardScaler()

    train_array = scaler.fit_transform(train_df)
    test_array = scaler.transform(test_df)

    # Create a sliding windowed data
    # Few constants like input window of number of time steps width and prediction timesteps width
    # Width = timesteps +1 because need to include the target at T
    window = WindowGenerator(input_width=(timesteps+1), label_width=1, shift=1,
                             input_columns=input_columns, label_columns=target_columns, all_columns=train_df.columns)
    td = window.make_dataset(train_array, True, 250)
    train_data = td.take(3)
    val_data = td.skip(3)

    # Save model weights
    ac_checkpoint_path = "checkpoint/autoencoder.ckpt"
    # Create a callback that saves the model's weights
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ac_checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
    
    train = tf.concat([x for x, _ in train_data], axis=0)
    val = tf.concat([x for x, _ in val_data], axis=0)

    # Train auto encoder model
    # Time steps +1 because of the same reason as when defining window
    AC = Autoencoder(num_timesteps=(timesteps+1), num_inputs=len(
        input_columns), num_hidden=2, kernel_size=25, pooling=pooling)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.1, decay_steps=50, decay_rate=0.97, staircase=True)
    AC.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(
        learning_rate=lr_schedule))
    AC.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, mode='min')
    # If model has been trained, reuse weights else train the model
    if os.path.isfile(os.getcwd()+'/'+ac_checkpoint_path+'.index'):
        AC.load_weights(ac_checkpoint_path)
    else:
        AC_model = AC.fit(x=train, y=train, validation_data=(val, val), epochs=100, callbacks=[
            early_stopping, checkpoint_callback])
        AC.summary()
        fig, axs = plt.subplots()
        axs.plot(AC_model.history['loss'])
        axs.plot(AC_model.history['val_loss'])
        axs.legend(['training loss', 'validation loss'])
    
    # Graph all normalized and autoencoded variables used in forecasting
    graph_normalized_and_autoencoded()
    
    
    # Get auto encoded data for training and validating with RNN/CNN model
    autoencoded_train_inputs = AC.predict(train)
    autoencoded_val_inputs = AC.predict(val)
    
    # Separate x_train, x_val and y_train from autoecoded data
    x_train = tf.convert_to_tensor(
        [x[:timesteps] for x in autoencoded_train_inputs])
    x_val = tf.convert_to_tensor([x[:timesteps]
                                 for x in autoencoded_val_inputs])
    y_train = tf.convert_to_tensor(
        [y[timesteps][-1] for y in autoencoded_train_inputs])
    y_val = tf.convert_to_tensor([[[y[timesteps][-1]]] for y in val])


    # Combine into a dataset to train the model
    model_train_dataset = tf.data.Dataset.from_tensors((x_train, y_train))
    model_val_dataset = tf.data.Dataset.from_tensors((x_val, y_val))

    # Define learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=300, decay_rate=0.95, staircase=True)
    # Define early stopping condition for model
    early_stopping = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=30, mode='min')

    # define sliding window
    lf = 1      # look forward
    ks = 2     # kernel size
    lw = 1      # label width
    lb = timesteps
    through_cnn = True
    # Train with RNN
    if through_cnn:
        cnn_checkpoint_path = "checkpoint/cnn.ckpt"
        # Create a callback that saves the model's weights
        cnn_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cnn_checkpoint_path, save_weights_only=True,verbose=1)          

        # Build CNN model                                               
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=ks, activation='relu', use_bias=False, input_shape=(timesteps, len(input_columns))))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2,))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=1))

        # Define loss function, optimizer
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(lr_schedule))
        model.run_eagerly = False

        # If model has been trained, reuse weights else train the model
        if os.path.isfile(os.getcwd()+'/'+cnn_checkpoint_path+'.index'):
            model.load_weights(cnn_checkpoint_path)
        else:
            model_history = model.fit(train_data=model_train_dataset, validation_data=model_val_dataset, epochs=300, callbacks=[cnn_checkpoint_callback, early_stopping])
            print(model.summary())

            # Plot loss from CNN
            fig, axs = plt.subplots()
            axs.plot(model_history.history['loss'])
            axs.plot(model_history.history['val_loss'])
            axs.legend(['training loss', 'validation loss'])
            plt.show()

    else:
        rnn_checkpoint_path = "checkpoint/rnn.ckpt"
        # Create a callback that saves the model's weights
        rnn_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=rnn_checkpoint_path, save_weights_only=True,verbose=1)

        # Build RNN model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(64, return_sequences=False,return_state=False, activation=None, use_bias=False))
        model.add(tf.keras.layers.Dense(64, activation=None, use_bias=False))
        model.add(tf.keras.layers.Dropout(0.01))
        model.add(tf.keras.layers.Dense(1, activation=None, use_bias=False))

        # Define loss function , optimizer
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(lr_schedule))
        model.run_eagerly = False

        # If model has been trained, reuse weights else train the model
        if os.path.isfile(os.getcwd()+'/'+rnn_checkpoint_path+'.index'):
            model.load_weights(rnn_checkpoint_path)
        else:
            model_history = model.fit(train_data=model_train_dataset, validation_data=model_val_dataset, epochs=300, callbacks=[early_stopping, rnn_checkpoint_callback])
            print(model.summary())

            # Plot loss from RNN
            fig, axs = plt.subplots()
            axs.plot(model_history.history['loss'])
            axs.plot(model_history.history['val_loss'])
            axs.legend(['training loss', 'validation loss'])
            plt.show()


    prediction_column = '_MKT'
    prediction_column_index = train_df.columns.get_loc(prediction_column)
    actual_columns = train_df.columns

    eval_train = window.make_dataset(train_array, shuffle=False)
    eval_test = window.make_dataset(test_array, shuffle=False)

    convertable_train_array = np.copy(train_array)

    convertable_test_array = np.copy(test_array)

    unscaled_train_array = scaler.inverse_transform(train_array)
    unscaled_test_array = scaler.inverse_transform(test_array)

    train_df = convert_and_add_phases(unscaled_train_array,actual_columns, train_df.index)
    test_df = convert_and_add_phases(unscaled_test_array,actual_columns,test_df.index)

    # CHECK Overall Model performance on Train Data
    plt.figure()
    plt.subplot()
    normalized_y_pred = model.predict(eval_train)
    convertable_train_array[timesteps+1:,prediction_column_index] = np.squeeze(normalized_y_pred)
    y_pred = convert_and_add_phases(scaler.inverse_transform(convertable_train_array),actual_columns, train_df.index).iloc[timesteps+1:,prediction_column_index]
    y_true = train_df[timesteps+1:].iloc[:,prediction_column_index]
    insample_mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
    plt.plot(y_true)
    plt.plot(y_pred, '--')
    plt.title('in-sample mse =%1.2f' % insample_mse)
    plt.legend(['y_true', 'y_pred'])

    plt.figure()
    plt.subplot()
    y_mkt = train_df.iloc[lb+lf:, prediction_column_index]
    # position taking: Directional trading strategy
    y_pred = np.squeeze(y_pred)
    pos = np.sign(y_pred)
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + pnl[:-1])))
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + y_mkt[:-1])))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('in-sample Sharpe ratio w/o threshold = %1.2f' % sr)
    plt.legend(['pnl [t+1]', 'underlying'])

    plt.figure()
    plt.subplot()    
    # position taking: Directional trading strategy with threshold
    y_mkt = train_df.iloc[lb+lf:, prediction_column_index]
    pos = np.sign(np.array([(lambda x: x if abs(x) > np.sqrt(insample_mse) else -abs(x))(x) for x in y_pred]))
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + pnl)))
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + y_mkt[:-1])))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('in-sample Sharpe ratio with threshold = %1.2f' % sr)
    plt.legend(['pnl [t+1]', 'underlying'])

    ### ERROR HERE
    plt.figure()
    plt.subplot()
    normalized_y_pred = model.predict(eval_test)
    convertable_test_array[timesteps+1:,18] = np.squeeze(normalized_y_pred)
    y_pred = convert_and_add_phases(scaler.inverse_transform(convertable_test_array),actual_columns, test_df.index).iloc[timesteps+1:,prediction_column_index]
    y_true = test_df.iloc[timesteps+1:,prediction_column_index]
    mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
    plt.plot(y_true[:, -1, -1])
    plt.plot(y_pred, '--')
    plt.title('out-of-sample mse =%1.2f' % mse)
    plt.legend(['y_true', 'y_pred'])

    plt.figure()
    plt.subplot()
    y_mkt = test_df.iloc[lb+lf:, :].loc[:, prediction_column]
    # position taking: Directional trading strategy with in-sample mse sqrt as threshold
    y_pred = np.squeeze(y_pred)
    pos = np.sign(y_pred)
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + pnl[:-1])))
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + y_mkt[:-1])))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('out-of-sample Sharpe ratio w/o threshold = %1.2f' % sr)
    plt.legend(['pnl [t+1]', 'underlying'])

    plt.figure()
    plt.subplot()
    # position taking: Directional trading strategy with in-sample mse sqrt as threshold
    pos = np.sign(np.array([(lambda x: x if abs(x) > np.sqrt(insample_mse) else -x)(x) for x in y_pred]))
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + pnl)))
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + y_mkt[:-1])))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('out-of-sample Sharpe ratio with threshold = %1.2f' % sr)
    plt.legend(['pnl [t+1]', 'underlying'])



    # Finding the threshold for each phase variable that model perform the best
    evaluating_test = test_df.iloc[lb+lf:, :].reset_index(drop=True)
    normalized_y_pred = model.predict(eval_train)
    convertable_train_array[timesteps+1:,18] = np.squeeze(normalized_y_pred)
    y_pred = convert_and_add_phases(scaler.inverse_transform(convertable_test_array),actual_columns, test_df.index).iloc[timesteps+1:,prediction_column_index]
    y_true = test_df.iloc[timesteps+1:,prediction_column_index]
    for variable in ('M2_phase','_OIL_phase','VIX_phase'):
        low_phased_index = list(evaluating_test[evaluating_test[variable]=='low'].index.values)
        medium_phased_index = list(evaluating_test[evaluating_test[variable]=='medium'].index.values)
        high_phased_index = list(evaluating_test[evaluating_test[variable]=='high'].index.values)
        print(variable+" MSE: ")
        print("Low = %f" %tf.reduce_mean(tf.keras.losses.MSE(y_true[low_phased_index], y_pred[low_phased_index])))
        print("Normal = %f" %tf.reduce_mean(tf.keras.losses.MSE(y_true[medium_phased_index], y_pred[medium_phased_index])))
        print("High = %f" %tf.reduce_mean(tf.keras.losses.MSE(y_true[high_phased_index], y_pred[high_phased_index])))
        print()

    # Trading strategy with thresholds from phase variables
    plt.figure()
    plt.subplot()
    # position taking: Directional trading strategy with in-sample mse sqrt as threshold
    y_pred = np.squeeze(y_pred[:,  -1])
    pos = np.sign(np.array([(lambda x: x if abs(x) > np.sqrt(insample_mse) else -x)(x) for x in y_pred]))
    pos[pos == -1] = 0
    pos[[i for i in range(pos.shape[0]) if i not in phased_index]] = 0
    pnl = pos[1:] * y_mkt[:-1]
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + pnl)))
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + y_mkt[:-1])))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('out-of-sample Sharpe ratio with threshold with VIX phase = %1.2f' % sr)
    plt.legend(['pnl [t+1]',  'underlying'])

    plt.figure()
    plt.subplot()
    evaluating_test = test_df.iloc[lb+lf:, :].reset_index(drop=True)
    phased_index = list(evaluating_test[evaluating_test['M2_phase']=='low'].index.values)
    y_mkt =  test_df.iloc[lb+lf:, :].loc[:,'_MKT']
    y_pred = model.predict(eval_test)
   
    # position taking: Directional trading strategy with in-sample mse sqrt as threshold
    y_pred = np.squeeze(y_pred[:,  -1])
    pos = np.sign(np.array([(lambda x: x if abs(x) > np.sqrt(insample_mse) else -x)(x) for x in y_pred]))
    pos[pos == -1] = 0
    pos[[i for i in range(pos.shape[0]) if i not in phased_index]] = 0
    pnl = pos[1:] * y_mkt[:-1]
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + pnl)))
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + y_mkt[:-1])))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('out-of-sample Sharpe ratio with threshold with M2 phase = %1.2f' % sr)
    plt.legend(['pnl [t+1]', 'underlying'])

    plt.figure()
    plt.subplot()
    evaluating_test = test_df.iloc[lb+lf:, :].reset_index(drop=True)
    phased_index = list(evaluating_test[evaluating_test['_OIL_phase']=='low'].index.values)
    y_mkt =  test_df.iloc[lb+lf:, :].loc[:,'_MKT']
    y_pred = model.predict(eval_test)
   
    # position taking: Directional trading strategy with in-sample mse sqrt as threshold
    y_pred = np.squeeze(y_pred[:,  -1])
    pos = np.sign(np.array([(lambda x: x if abs(x) > np.sqrt(insample_mse) else -x)(x) for x in y_pred]))
    pos[pos == -1] = 0
    pos[[i for i in range(pos.shape[0]) if i not in phased_index]] = 0
    pnl = pos[1:] * y_mkt[:-1]
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + pnl)))
    plt.plot(y_mkt.index[:-1], np.cumprod((np.ones(y_mkt[:-1].shape) + y_mkt[:-1])))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('out-of-sample Sharpe ratio with threshold with OIL phase = %1.2f' % sr)
    plt.legend(['pnl [t+1]',  'underlying'])
    plt.show()