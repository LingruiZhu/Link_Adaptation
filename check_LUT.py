import numpy as np
import h5py

bler_file = h5py.File("BLER_LUT_data_simulation/LUT_AWGN_simulation_customized.h5", "r")
print(bler_file.keys())
sinr_mcs_bler_array = np.array(bler_file["sinr_mcs_bler_array"])
sinr_list = list(bler_file.get("sinr_list"))
print(np.shape(sinr_mcs_bler_array))