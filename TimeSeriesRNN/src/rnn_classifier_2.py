import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from sklearn.model_selection import TimeSeriesSplit

def main():
    EDA()

    settings = ['elman', 'jordan', 'multi']
    for setting in settings:
        cross_validation(setting)
        print(setting + " Done!")

def cross_validation(setting):

    # The five different datasets
    # delhi_df = pd.read_csv('data/DailyDelhiClimateTrain.csv')
    # data = delhi_df['meantemp'].values
    # result_file = open("Results/delhi_" + setting + ".txt", "w")

    # ibm_df = pd.read_csv('data/IBM.csv')
    # data = ibm_df['Close'].values
    # result_file = open("Results/ibm_" + setting + ".txt", "w")

    # google_df = pd.read_csv('data/google.csv')
    # data = google_df['Close'].values[0:1000]
    # result_file = open("Results/google_" + setting + ".txt", "w")

    # elec_df = pd.read_csv('data/Electric_Production.csv')
    # data = elec_df['Value'].values
    # result_file = open("Results/elec_" + setting + ".txt", "w")

    corn_df = pd.read_csv('data/corn.txt')
    data = corn_df.iloc[:, 1].values
    result_file = open("Results/corn_" + setting + ".txt", "w")

    num_instances = len(data)
    train_perc = 0.8
    train_size = int(num_instances*train_perc)
    data_train = data[0:train_size]
    data_test = data[train_size:]

    tscv = TimeSeriesSplit(n_splits = 10)

    start_hidden_units = 10
    start_epochs = 2
    start_seq_len = 2
    start_learning_rate = 0.001

    best_avg_mse = 1000000
    best_com = [0, 0, 0, 0]

    hidden_units = start_hidden_units
    epochs = start_epochs
    seq_len = start_seq_len
    learning_rate = start_learning_rate

    while hidden_units <= 100:
        epochs = start_epochs
        while epochs <= 20:
            seq_len = start_seq_len
            while seq_len <= 20:
                learning_rate = start_learning_rate
                while learning_rate <= 0.013:
                    mse_values = []
                    for train_index, test_index in tscv.split(data_train):
                        cv_train, cv_test = data_train[train_index], data_train[test_index]
                        X_train = cv_train[:-1]
                        Y_train = cv_train[1:]
                        if setting == 'elman':
                            model = elman_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate)
                        elif setting == 'jordan':
                            model = jordan_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate)
                        elif setting == 'multi':
                            model = multi_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate)
                        X_test = cv_test[0:-1]
                        Y_true = cv_test[1:]
                        results = model.evaluate(X_test, Y_true, batch_size = 1)
                        mse_values.append(results)

                    avg_mse = sum(mse_values)/len(mse_values)
                    if avg_mse < best_avg_mse:
                            best_avg_mse = avg_mse
                            best_com = [hidden_units, epochs, seq_len, learning_rate]
                learning_rate += 0.002
                seq_len += 3
            epochs += 3
        hidden_units += 10
    print(best_avg_mse)
    print(best_com)

    hidden_units, epochs, seq_len, learning_rate = best_com

    X_train = data_train[:-1]
    Y_train = data_train[1:]
    X_test = data_test[:-1]
    Y_test = data_test[1:]

    num_runs = 20
    mse_values = []
    for i in range(num_runs):
        if setting == 'elman':
            model = elman_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate)
        elif setting == 'jordan':
            model = jordan_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate)
        elif setting == 'multi':
            model = multi_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate)
        results = model.evaluate(X_test, Y_test, batch_size = 1)
        mse_values.append(results)
    
    print('Algorithm: ', setting, file = result_file)
    print('Mean: ', np.mean(mse_values), file = result_file)
    print('Std: ', np.std(mse_values), file = result_file)


# Elman RNN: Feed previous hidden layer states as input to current hidden layer
def elman_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate):
    input = Input(shape = (1,))
    output = elman_combined_layer(hidden_units, seq_len)(input)

    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate), loss = tf.keras.losses.MeanSquaredError())
    print(model.summary())
    model.fit(X_train, Y_train, epochs = epochs, batch_size = 1)
    return model

class elman_combined_layer(Layer):
    def __init__(self, units = 32, num_context_layers = 1):
        self.units = units
        self.num_context_layers = num_context_layers
        self.context = tf.Variable(tf.zeros(shape = (self.num_context_layers, self.units), dtype = tf.float32), trainable=False)
        super(elman_combined_layer, self).__init__()

    def build(self, input_shape):
        self.input_W = self.add_weight(name='input_weight', shape = (input_shape[-1], self.units), 
                               initializer='random_normal', trainable=True)
        self.context_W = self.add_weight(name='context_weight', shape = (self.num_context_layers, self.units, self.units), 
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='bias', shape = (1, self.units), 
                               initializer='zeros', trainable=True) 
        self.output_W = self.add_weight(name='output_weight', shape = (self.units, 1), 
                               initializer='random_normal', trainable=True)
        self.output_b = self.add_weight(name='output_bias',  shape = (1,),
                               initializer='zeros', trainable=True) 
        super(elman_combined_layer, self).build(input_shape)

    def call(self, x):
        input = tf.matmul(x, self.input_W)
        context = tf.keras.backend.batch_dot(self.context, self.context_W)
        context = tf.reduce_sum(context, 0)
        net_input = input + self.b + context
        hidden_output = tf.keras.activations.sigmoid(net_input)
        context_layers = tf.concat([tf.reshape(hidden_output[0], shape = (1, self.input_W.shape[1])), self.context], axis = 0)
        context_layers = context_layers[0:-1]
        self.context.assign(context_layers)
        final_output = tf.matmul(hidden_output, self.output_W) + self.output_b
        return final_output

# Jordan RNN: Feed previous output layer states as input to current hidden layer
def jordan_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate):
    input = Input(shape = (1,))
    output = jordon_combined_layer(hidden_units, seq_len)(input)
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate), loss = tf.keras.losses.MeanSquaredError())
    print(model.summary())
    model.fit(X_train, Y_train, epochs = epochs, batch_size = 1)
    return model

class jordon_combined_layer(Layer):
    def __init__(self, units = 32, num_state_layers = 1):
        self.units = units
        self.num_state_layers = num_state_layers
        self.state = tf.Variable(tf.zeros(shape = (self.num_state_layers, 1), dtype = tf.float32), trainable=False)
        super(jordon_combined_layer, self).__init__()

    def build(self, input_shape):
        self.input_W = self.add_weight(name='input_weight', shape = (input_shape[-1], self.units), 
                               initializer='random_normal', trainable=True)
        self.state_W = self.add_weight(name='state_weight', shape = (self.num_state_layers, 1, self.units), 
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='bias', shape = (1, self.units), 
                               initializer='zeros', trainable=True) 
        self.output_W = self.add_weight(name='output_weight', shape = (self.units, 1), 
                               initializer='random_normal', trainable=True)
        self.output_b = self.add_weight(name='output_bias',  shape = (1,),
                               initializer='zeros', trainable=True) 
        super(jordon_combined_layer, self).build(input_shape)

    def call(self, x):
        input = tf.matmul(x, self.input_W)
        state = tf.keras.backend.batch_dot(self.state, self.state_W)
        state = tf.reduce_sum(state, 0)
        net_input = input + self.b + state
        hidden_output = tf.keras.activations.sigmoid(net_input)
        final_output = tf.matmul(hidden_output, self.output_W) + self.output_b
        state_layers = tf.concat([final_output, self.state], axis = 0)
        state_layers = state_layers[0:-1]
        self.state.assign(state_layers)
        return final_output

# Multi RNN: Feed previous hidden and output layer states as input to current hidden layer
def multi_rnn(X_train, Y_train, hidden_units, epochs, seq_len, learning_rate):
    input = Input(shape = (1,))
    output = multi_combined_layer(hidden_units, seq_len)(input)
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate), loss = tf.keras.losses.MeanSquaredError())
    print(model.summary())
    model.fit(X_train, Y_train, epochs = epochs, batch_size = 1)
    return model

class multi_combined_layer(Layer):
    def __init__(self, units = 32, num_context_layers = 1):
        self.units = units
        self.num_context_layers = num_context_layers
        self.context = tf.Variable(tf.zeros(shape = (self.num_context_layers, self.units), dtype = tf.float32), trainable=False)
        self.state = tf.Variable(tf.zeros(shape = (self.num_context_layers, 1), dtype = tf.float32), trainable=False)
        super(multi_combined_layer, self).__init__()

    def build(self, input_shape):
        self.input_W = self.add_weight(name='input_weight', shape = (input_shape[-1], self.units), 
                               initializer='random_normal', trainable=True)
        self.context_W = self.add_weight(name='context_weight', shape = (self.num_context_layers, self.units, self.units), 
                               initializer='random_normal', trainable=True)
        self.state_W = self.add_weight(name='state_weight', shape = (self.num_context_layers, 1, self.units), 
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='bias', shape = (1, self.units), 
                               initializer='zeros', trainable=True) 
        self.output_W = self.add_weight(name='output_weight', shape = (self.units, 1), 
                               initializer='random_normal', trainable=True)
        self.output_b = self.add_weight(name='output_bias',  shape = (1,),
                               initializer='zeros', trainable=True) 
        super(multi_combined_layer, self).build(input_shape)

    def call(self, x):
        input = tf.matmul(x, self.input_W)
        context = tf.keras.backend.batch_dot(self.context, self.context_W)
        context = tf.reduce_sum(context, 0)
        state = tf.keras.backend.batch_dot(self.state, self.state_W)
        state = tf.reduce_sum(state, 0)
        net_input = input + self.b + context + state
        hidden_output = tf.keras.activations.sigmoid(net_input)
        context_layers = tf.concat([tf.reshape(hidden_output[0], shape = (1, self.input_W.shape[1])), self.context], axis = 0)
        context_layers = context_layers[0:-1]
        self.context.assign(context_layers)
        final_output = tf.matmul(hidden_output, self.output_W) + self.output_b
        state_layers = tf.concat([final_output, self.state], axis = 0)
        state_layers = state_layers[0:-1]
        self.state.assign(state_layers)
        return final_output

# Perform EDA on the 5 datasets
def EDA():
    delhi_df = pd.read_csv('data/DailyDelhiClimateTrain.csv')
    data = delhi_df['meantemp'].values
    plt.plot(data, 'b')
    plt.title('New Delhi mean temperatures over time')
    plt.xlabel('Data point index')
    plt.ylabel('Mean temperature')
    plt.savefig('Results/new_delhi_plot.png')
    plt.clf()

    ibm_df = pd.read_csv('data/IBM.csv')
    data = ibm_df['Close'].values
    plt.plot(data, 'b')
    plt.title('IBM stock prices over time')
    plt.xlabel('Data point index')
    plt.ylabel('IBM stock price')
    plt.savefig('Results/ibm_stock_plot.png')
    plt.clf()

    google_df = pd.read_csv('data/google.csv')
    data = google_df['Close'].values
    plt.plot(data, 'b')
    plt.title('Google stock prices over time')
    plt.xlabel('Data point index')
    plt.ylabel('Google stock price')
    plt.savefig('Results/google_stock_plot.png')
    plt.clf()

    elec_df = pd.read_csv('data/Electric_Production.csv')
    data = elec_df['Value'].values
    plt.plot(data, 'b')
    plt.title('Electricity consumption over time')
    plt.xlabel('Data point index')
    plt.ylabel('Electricity consumption')
    plt.savefig('Results/elec_cons_plot.png')
    plt.clf()

    corn_df = pd.read_csv('data/corn.txt')
    data = corn_df.iloc[:, 1].values
    plt.plot(data, 'b')
    plt.title('Corn price over time')
    plt.xlabel('Data point index')
    plt.ylabel('Corn price')
    plt.savefig('Results/corn_plot.png')
    plt.clf()

if __name__ == '__main__':
    main()