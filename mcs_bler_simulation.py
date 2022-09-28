from operator import mod
from sionna.ofdm import ResourceGrid
import numpy as np
import math
import h5py

from LinkSimulation import Link_Simulation
from Simulation_Parameters import Simulation_Parameter, Channel_Model
from MCS_and_CQI import ModulationCodingScheme, ChannelQualityIndex, get_MCS, get_CQI


def get_default_rg():
    """this funtion returns a default resourse grid with pilot, data and guard carriers
    """
    resouce_grid = ResourceGrid(num_ofdm_symbols=14,
                                fft_size=72,
                                subcarrier_spacing=30e3,
                                num_tx=1,
                                num_streams_per_tx=1,
                                cyclic_prefix_length=6,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=[2, 11],
                                num_guard_carriers = [5, 6],
                                dc_null=True
                                )
    print(resouce_grid.num_data_symbols)
    return resouce_grid


def single_MCS_simulation(modulation_order:int, code_rate:float, default_sim_paras:Simulation_Parameter, ebno_db:float):
    default_sim_paras.set_num_bits_per_symbol(modulation_order)
    default_sim_paras.set_code_rate(code_rate)

    link_simulation = Link_Simulation(default_sim_paras)
    _, bler = link_simulation.run(ebno_db)
    return bler


def single_SNR_simulation(default_sim_paras:Simulation_Parameter, mcs:ModulationCodingScheme, ebno_db:float)->dict:
    num_bits_per_symbol_list = mcs.number_bits_per_symbol
    code_rates_list = mcs.code_rate
    mcs_indicies = mcs.mcs_index

    bler_dict_single_sinr = dict()
    bler_list_single_sinr = list()
    for num_bits_per_symbol, code_rate, mcs_index in zip(num_bits_per_symbol_list, code_rates_list, mcs_indicies):
        bler = single_MCS_simulation(num_bits_per_symbol, code_rate, default_sim_paras, ebno_db)
        bler_dict_single_sinr[mcs_index] = bler
        bler_list_single_sinr.append(bler)
    return bler_dict_single_sinr, bler_list_single_sinr


def simulate_BLER_for_mcs():
    # set a default simulation parameters settings
    resouce_grid = get_default_rg()
    batch_size = 1000
    num_bits_per_symbol = 4
    code_rate = 0
    carrier_frequency = 2.6e9
    ue_speed = 0
    delay_spread = 100e-9
    channel_model = Channel_Model.AWGN

    sim_para_default = Simulation_Parameter(resouce_grid, batch_size, num_bits_per_symbol, code_rate, carrier_frequency, ue_speed, delay_spread, channel_model)
    
    # get CQI and MCS
    cqi_sinr = get_CQI()
    sinr_list = cqi_sinr.sinr_list
    cqi_code_list = cqi_sinr.cqi_list
    mcs = get_MCS()
    sinr_mcs_bler_dict = dict()
    sinr_mcs_bler_list = list()

    # first part to save in .h5 file
    bler_file = h5py.File("BLER_LUT_data_MCS/table3_LUT_AWGN_mcs.h5", "w")
    cqi_group = bler_file.create_group("cqi_sinr_table")
    cqi_group.create_dataset("cqi_code", data=cqi_code_list)
    cqi_group.create_dataset("sinr_list", data=sinr_list)

    for sinr, cqi_code in zip(sinr_list, cqi_code_list):
        print("currently running the simulation for CQI ... {}".format(cqi_code))
        bler_dict_single_sinr, bler_list_single_sinr = single_SNR_simulation(sim_para_default, mcs, sinr)
        sinr_mcs_bler_dict[cqi_code] = bler_dict_single_sinr
        sinr_mcs_bler_list.append(bler_list_single_sinr)
    np.save("BLER_LUT_data_MCS/table3_LUT_AWGN_channel_mcs.npy", sinr_mcs_bler_dict)

    # second part to save in h5 file
    sinr_mcs_bler_array = np.array(sinr_mcs_bler_list)
    bler_file.create_dataset(name="sinr_mcs_bler_array", data=sinr_mcs_bler_array)
    bler_file.close()


def simulate_BLER_for_simulation():
    resource_grid = get_default_rg()
    batch_size = 1000
    num_bits_per_symbol = 4
    code_rate = 0
    carrier_frequency = 2.6e9
    ue_speed = 0
    delay_spread = 100e-9
    channel_model = Channel_Model.AWGN

    sim_para_default = Simulation_Parameter(resource_grid, batch_size, num_bits_per_symbol, code_rate, carrier_frequency, \
        ue_speed, delay_spread, channel_model)
    
    # get CQI and MCS
    sinr_list = np.arange(-10, 30, 0.5)
    mcs = get_MCS()
    sinr_mcs_bler_dict = dict()
    sinr_mcs_bler_list = list()

    # save sinr list to .h5 file
    bler_file = h5py.File("BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5", "w")
    bler_file.create_dataset("sinr_list", data=sinr_list)

    for sinr in sinr_list:
        print("currently running the simulation for sinr ... {} dB".format(sinr))
        bler_dict_single_sinr, bler_list_single_sinr = single_SNR_simulation(sim_para_default, mcs, sinr)
        sinr_mcs_bler_dict[sinr] = bler_dict_single_sinr
        sinr_mcs_bler_list.append(bler_list_single_sinr)
    # save result to .npy file
    np.save("BLER_LUT_data_simulation/table3_LUT_AWGN_channel_full_sinr.npy", sinr_mcs_bler_dict)
    # save result to .h5 file
    sinr_mcs_bler_array = np.array(sinr_mcs_bler_list)
    bler_file.create_dataset("sinr_mcs_bler_array", data=sinr_mcs_bler_array)
    bler_file.close()


if __name__ == "__main__":
    simulate_BLER_for_mcs()
    simulate_BLER_for_simulation()