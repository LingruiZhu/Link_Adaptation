from sklearn.linear_model import LogisticRegression, LinearRegression
import h5py
import matplotlib.pyplot as plt
import numpy as np


def modify_zero_and_one(array_a: np.array):
    a_list = array_a.tolist()
    new_list = list()
    for element in a_list:
        if element == 1:
            element = element - 1e-8
        elif element == 0:
            element = element + 1e-8
        new_list.append(element)
    new_array = np.array(new_list)
    return new_array


def sigmoid(alpha, beta, x:np.array):
    y = 1 / (1 + np.exp(alpha + beta*x))
    return y


BLER_file = "/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5"
BLER_data = h5py.File(BLER_file, "r") 
sinr_mcs_table = BLER_data["sinr_mcs_bler_array"]   # shape (80, 20) 80->number of 
sinr_list = list(BLER_data["sinr_list"][:])


log_reg = LogisticRegression()
lin_reg = LinearRegression()
x = np.expand_dims(np.array(sinr_list), axis=-1)
y = np.array(sinr_mcs_table[:,10])
y_new = modify_zero_and_one(y)
y_log_feature = np.log((1/y_new) - 1)

x_train = x[33:38]
y_train = y_log_feature[33:38]

lin_reg.fit(x_train, y_train)
beta = lin_reg.coef_
alpha = lin_reg.intercept_
x_predict = np.array(sinr_list)

y_predict = sigmoid(alpha, beta, x_predict)

plt.figure()
plt.plot(x, y_log_feature, "b-x", label="train data")
plt.plot(x_predict, alpha+beta*x_predict, "r-o", label="prediction")
plt.grid()
plt.legend()

plt.figure()
i = 10
for i in range(20):
    plt.plot(sinr_list, sinr_mcs_table[:,i], "-x", label="MCS "+str(i))
plt.grid()
plt.legend()
plt.xlabel("SNR")
plt.ylabel("BLER")
plt.show()