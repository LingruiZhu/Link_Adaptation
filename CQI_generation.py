from cmath import sin
import matplotlib.pyplot as plt
import numpy as np
import h5py

from link_envinorment import LutEnvironment
from InterferenceToy import InterferenceToy
from quantization import UniformQuantizer


def generate_single_trail_data_time(num_time_steps:int, file_path:str=None, plot_option:bool=False, quantize_sinr:bool=False):
    link_env = LutEnvironment(ebno_db=10.5)
    interference1 = InterferenceToy(tx_power=0.1, h_factor=0.3, interval=10, period=2)
    interference2 = InterferenceToy(tx_power=0.1, h_factor=0.2, interval=6, period=3)
    interference3 = InterferenceToy(tx_power=0.1, h_factor=0.3, interval=10, period=1)
    link_env.add_interference_noise(interference1)
    link_env.add_interference_noise(interference2)
    link_env.add_interference_noise(interference3)
    link_env.set_interference_mode("time")

    num_bits_per_symbol = 4
    code_rate = 553/1024
    cqi_list = list()
    sinr_list = list()
    for i in range(num_time_steps):
        _, _, cqi, sinr = link_env.step(num_bits_per_symbol, code_rate)
        cqi_list.append(cqi)
        sinr_list.append(sinr)
    
    if quantize_sinr:
        sinr_quantized = quantize_sinr(sinr_list)

    if plot_option:
        plt.figure()
        plt.plot(np.arange(num_time_steps), sinr_list, "-x")
        if quantize_sinr:
            plt.plot(np.arange(num_time_steps), sinr_quantized, "r-o")
        plt.grid()
        plt.show()
    
    if file_path is not None:
        if quantize_sinr:
            save_quantized_sinr(sinr_list, sinr_quantized, cqi_list, file_path)
        else:
            save_sinr(sinr_list, cqi_list, file_path)


def generate_data_interference_space(num_time_steps:int, file_path:str=None, plot_option:bool=False, is_sinr_quatized:bool=False):
    link_env = LutEnvironment(ebno_db=10.5)
    interference1 = InterferenceToy(tx_power=0.1, h_factor=0.4, area=[2, 8, 5, 15])
    interference2 = InterferenceToy(tx_power=0.1, h_factor=0.2, area=[5, 15, -15, -3])
    interference3 = InterferenceToy(tx_power=0.1, h_factor=0.3, area=[-6, -2, -15, -4])
    interference4 = InterferenceToy(tx_power=0.1, h_factor=0.1, area=[-15, -8, 0, 15])
    
    link_env.set_interference_mode("space")
    link_env.add_interference_noise(interference1)
    link_env.add_interference_noise(interference2)
    link_env.add_interference_noise(interference3)
    link_env.add_interference_noise(interference4)

    num_bits_per_symbol = 4
    code_rate = 553/1024
    cqi_list = list()
    sinr_list = list()
    position_list = list()
    for i in range(num_time_steps):
        _, _, cqi, sinr, position = link_env.step(num_bits_per_symbol, code_rate)
        cqi_list.append(cqi)
        sinr_list.append(sinr)
        position_list.append(position)

    if is_sinr_quatized:
        sinr_quantized = quantize_sinr(sinr_list)

    if plot_option:
        plt.figure()
        plt.plot(np.arange(100), sinr_list[:100], "-x")
        if quantize_sinr:
            plt.plot(np.arange(100), sinr_quantized[:100], "r-o")
        plt.grid()
        plt.show()
    
    if file_path is not None:
        if quantize_sinr:
            save_quantized_sinr_positions(sinr_list, sinr_quantized, cqi_list, position_list, file_path)
        else:
            raise ValueError("I hope you can use quantized sinr, since unquantized is also included there :-)")


def quantize_sinr(sinr_list):
    quantizer = UniformQuantizer(min=5, max=12, num_bits=4)
    sinr_quant_list = list()
    for sinr in sinr_list:
        sinr_quant = quantizer(sinr)
        sinr_quant_list.append(sinr_quant)
    return sinr_quant_list


def save_sinr(sinr_list:list, cqi_list:list, file_path:str):
    h5file = h5py.File(file_path, "w")
    h5file.create_dataset(name="sinr", data=np.array(sinr_list))
    h5file.create_dataset(name="cqi", data=np.array(cqi_list))
    h5file.close()


def save_quantized_sinr(sinr_list:list, sinr_list_quantized:list, cqi_list:list, file_path:str):
    h5file = h5py.File(file_path, "w")
    h5file.create_dataset(name="sinr", data=np.array(sinr_list))
    h5file.create_dataset(name="cqi", data=np.array(cqi_list))
    h5file.create_dataset(name="quantized_sinr", data=np.array(sinr_list_quantized))
    h5file.close()


def save_quantized_sinr_positions(sinr_list:list, sinr_list_quantized:list, cqi_list:list, position_list:list, file_path:str):
    h5file = h5py.File(file_path, "w")
    h5file.create_dataset(name="sinr", data=np.array(sinr_list))
    h5file.create_dataset(name="cqi", data=np.array(cqi_list))
    h5file.create_dataset(name="quantized_sinr", data=np.array(sinr_list_quantized))
    h5file.create_dataset(name="positions", data=position_list)
    h5file.close()


if __name__ == "__main__":
    # h5_path = "interference_data/channel_interference_1_2_3.h5"
    # generate_single_trail_data(num_time_steps=100000, file_path=h5_path, plot_option=True, quantize_sinr=False)
    space_interference_file = "interference_space_data/inteference_4_bigger_areas.h5"
    generate_data_interference_space(num_time_steps=100000, plot_option=True, is_sinr_quatized=True, file_path=space_interference_file)