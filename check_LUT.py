import numpy as np

npy_file = "BLER_LUT_data/table3_LUT_CDL_channel.npy"
result_data = np.load(npy_file, allow_pickle=True)
result_data_dict = result_data.item()
