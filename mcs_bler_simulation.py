from operator import mod
from sionna.ofdm import ResourceGrid
import numpy as np

from baseband_processing import Link_Simulation
from Simulation_Parameters import Simulation_Parameter
from MCS_and_CQI import ModulationCodingScheme, ChannelQualityIndex

def get_MCS():
    # from 5.1.3.1-3 MCS index table 3 for PDSCH
    # TODO: also to update the other two tables
    modulation_order = [2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2,
                        4, 4, 4, 4, 4,
                        4, 6, 6, 6, 6,
                        6, 6, 6, 6]
    info_bits_length = [20, 40, 50, 64, 78,
                        99, 120, 157, 193, 251,
                        308, 379, 449, 526, 602,
                        340, 378, 434, 490, 553,
                        616, 438, 466, 517, 567,
                        616, 666, 719, 772]
    code_rate = [x/1024 for x in info_bits_length]
    mcs_index = [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28]
    mcs = ModulationCodingScheme(code_rate=code_rate, modulation_order=modulation_order, mcs_index=mcs_index)
    return mcs


def get_MCS_test():
    mod_order = [2, 4]
    code_rate = [0.3, 0.5]
    mcs_index = [0, 1]
    mcs_test = ModulationCodingScheme(code_rate=code_rate, modulation_order=mod_order, mcs_index=mcs_index)
    return mcs_test


def get_CQI():
    sinr = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7,
            14.1, 16.3, 18.7, 21.0, 22.7]
    cqi_code = [1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10,
                11, 12, 13, 14, 15]
    cqi = ChannelQualityIndex(cqi_code, sinr)
    return cqi


def get_CQI_test():
    sinr = [0, 5]
    cqi_code = [1, 2]
    cqi_test = ChannelQualityIndex(cqi_code, sinr)
    return cqi_test


def single_MCS_simulation(modulation_order:int, code_rate:float, default_sim_paras:Simulation_Parameter, ebno_db:float):
    default_sim_paras.set_num_bits_per_symbol(modulation_order)
    default_sim_paras.set_code_rate(code_rate)

    link_simulation = Link_Simulation(default_sim_paras)
    bler = link_simulation.run(ebno_db)
    return bler


def single_SNR_simulation(default_sim_paras:Simulation_Parameter, mcs:ModulationCodingScheme, ebno_db:float)->dict:
    modulation_order_list = mcs.modulation_order
    code_rates_list = mcs.code_rate
    mcs_indicies = mcs.mcs_index

    bler_dict_single_sinr = dict()
    for num_bits_per_symbol, code_rate, mcs_index in zip(modulation_order_list, code_rates_list, mcs_indicies):
        bler, ber = single_MCS_simulation(num_bits_per_symbol, code_rate, default_sim_paras, ebno_db)
        bler_dict_single_sinr[mcs_index] = bler
    return bler_dict_single_sinr


def main():
    # set a default simulation parameters settings
    resouce_grid = ResourceGrid(num_ofdm_symbols=14,
                                fft_size=76,
                                subcarrier_spacing=30e3,
                                num_tx=1,
                                num_streams_per_tx=1,
                                cyclic_prefix_length=6,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=[2, 11])
    batch_size = 10
    num_bits_per_symbol = 4
    code_rate = 0.5
    carrier_frequency = 2.6e9
    ue_speed = 0
    delay_spread = 100e-9#
    sim_para_default = Simulation_Parameter(resouce_grid, batch_size, num_bits_per_symbol, code_rate, carrier_frequency, ue_speed, delay_spread)
    
    # get CQI and MCS
    cqi_sinr = get_CQI()
    sinr_list = cqi_sinr.sinr_list
    cqi_code_list = cqi_sinr.cqi_list
    mcs = get_MCS()
    sinr_mcs_bler_dict = dict()

    for sinr, cqi_code in zip(sinr_list, cqi_code_list):
        print("currently running the simulation for CQI ... {}".format(cqi_code))
        bler_dict_single_sinr = single_SNR_simulation(sim_para_default, mcs, sinr)
        sinr_mcs_bler_dict[cqi_code] = bler_dict_single_sinr
    np.save("BLER_LUT_data/table3_LUT_CDL_channel.npy", sinr_mcs_bler_dict)


if __name__ == "__main__":
    main()
