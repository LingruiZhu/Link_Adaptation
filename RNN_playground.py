import matplotlib.pyplot as plt
import numpy as np
import time 
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.layers import LSTM


def get_data(file_path = "rnn_data/bike_rnn.csv", sequence_length=20):
    max_values = 45949
    with open(file_path) as f:
        data = csv.reader(f)
        next(data)
        bike = []
        nb_of_values = 0
        for line in data:
            bike.append(float(line[0]))
            nb_of_values += 1
            if nb_of_values >= max_values:
                break
    result = []
    for index in range(len(bike) - sequence_length):
        result.append(bike[index:index + sequence_length])
    result = np.array(result)

    result_mean = np.mean(result)
    result -= result_mean

    # divide test and training set
    row = int(round(0.9*result.shape[0]))
    train = result[:row, :]
    test = result[row:, :]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()
    layers = [1, 50, 100, 1]

    model.add(LSTM(units=layers[1], input_shape=(None,1), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model


def train_rnn():
    epochs = 30
    path_to_dataset = "rnn_data/bike_rnn.csv"
    sequence_length = 20

    X_train, y_train, X_test, y_test = get_data(path_to_dataset, sequence_length)
    model = build_model()
    model.summary()
    model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.05)
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))

    x_indicies = np.arange(np.size(y_test))

    plt.figure()
    plt.plot(x_indicies, predicted, "b-x", label="prediction")
    plt.plot(x_indicies, np.squeeze(y_test), "g-o", label="ground truth")
    plt.legend()
    plt.grid()
    plt.show()

    return model, y_test


def AR_model():
    ini_states = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    alpha = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
    state_lenght = 6
    time_steps = 1000
    ar_test = []
    for i in range(time_steps):
        new_state = 0
        new_state = np.sum([i*j for i,j in zip(alpha, ini_states)]) + np.random.normal(0, 0.01)
        ini_states.append(new_state)
        ini_states = ini_states[-state_lenght:]
        ar_test.append(new_state)
    
    plt.figure()
    plt.plot(np.arange(len(ar_test)), ar_test, "b-x")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    #train_rnn()
    AR_model()