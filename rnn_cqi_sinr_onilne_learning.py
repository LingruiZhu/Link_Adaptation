import numpy as np
import h5py
import matplotlib.pyplot as plt

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential


def read_data_set(keyword:str):
    file_path = "interference_data/channel_interference_1_2.h5"
    h5_file = h5py.File(file_path, "r")
    if keyword == "cqi":
        data = np.array(h5_file.get("cqi"))
    elif keyword == "sinr":
        data = np.array(h5_file.get("sinr"))
    return data


def cut_data_frame(data: np.array, window_size:int, num_input:int):
    num_time_instants = np.shape(data)[0]
    num_samples = num_time_instants - window_size + 1

    input_list = list()
    output_list = list()

    for i in range(num_samples):
        sliding_window = data[i:i+window_size]
        input_tmp  = sliding_window[:num_input]
        output_tmp = sliding_window[num_input:]
        input_list.append(input_tmp)
        output_list.append(output_tmp)
    
    samples_input = np.array(input_list)
    samples_output = np.array(output_list)
    samples_input = np.expand_dims(samples_input, axis=-1)
    samples_output = np.expand_dims(samples_output, axis=-1)
    return samples_input, samples_output



def build_rnn_model():
    """here to huild up a rnn model to predict time series of CQI or SINR
    """
    rnn_model = Sequential()
    rnn_model.add(LSTM(units=50, input_shape=(None, 1), return_sequences=True))
    rnn_model.add(Dropout(rate=0.1))
    rnn_model.add(LSTM(100, return_sequences=False))
    rnn_model.add(Dropout(rate=0.1))
    rnn_model.add(Dense(1, activation="linear"))
    rnn_model.compile(loss="mse", optimizer="rmsprop")
    return rnn_model


def train_rnn_model(num_updates:int=None, update_interval:int=None):
    """ use a online learning method to train RNN using the dataset
    """
    rnn_model = build_rnn_model()
    sinr_data = read_data_set(keyword="sinr")
    X, Y = cut_data_frame(sinr_data, window_size=20, num_input=19)
    print("=================== data has been taken ==========================")
    x_train, x_test = X[:20000, :], X[-2000:, :]
    y_train, y_test = Y[:20000, :], Y[-2000:, :]

    rnn_model.fit(x_train, y_train, batch_size=512, epochs=5)
    predicted = rnn_model.predict(x_test)
    mse = np.mean((predicted-y_test)**2)

    x_indices = np.arange(np.shape(y_test)[0])
    plt.figure()
    plt.plot(x_indices, predicted, "x-b", label="predicted")
    plt.plot(x_indices, np.squeeze(y_test), "o-g", label="ground truth")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    train_rnn_model()
