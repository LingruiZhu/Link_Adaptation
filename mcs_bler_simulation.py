from sionna.ofdm import ResourceGrid
import numpy as np
from baseband_processing import Link_Simulation
from Simulation_Parameters import Simulation_Parameter


def get_MCS():
    # from 5.1.3.1-3 MCS index table 3 for PDSCH
    # TODO: also to update the other two tables
    MCS = dict()
    MCS["modulation_order"] = [2, 2, 2, 2, 2,
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

    MCS["code_rate"] = [x/1024 for x in info_bits_length]
    MCS["MCS_index"] = [0, 1, 2, 4, 5, 6 ,7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27]
    return MCS


def get_CQI():
    CQI_SINR = dict()
    CQI_SINR["SINR"] = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7,
            14.1, 16.3, 18.7, 21.0, 22.7]
    CQI_SINR["cqi_code"] = [1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10,
                        11, 12, 13, 14, 15]
    return CQI_SINR


def single_MCS_simulation(modulation_order:int, code_rate:float, default_sim_paras:Simulation_Parameter, ebno_db:float):
    default_sim_paras.set_num_bits_per_symbol(modulation_order)
    default_sim_paras.set_code_rate(code_rate)

    link_simulation = Link_Simulation(default_sim_paras)
    bler = link_simulation.run(ebno_db)
    return bler


def single_SNR_simulation(default_sim_paras:Simulation_Parameter, ebno_db:float):
    mcs = get_MCS()
    modulation_order_list = mcs["modulation_order"]
    code_rates_list = mcs["code_rate"]
    mcs_indicies = mcs["MCS_index"]

    # remove first 9 mcs since coderate < 1/5 is not supported for Sionna yet.
    modulation_order_list_clipped = modulation_order_list[9:]
    code_rates_list_cliped = code_rates_list[9:]
    mcs_indicies_cliped = mcs_indicies[9:]

    bler_dict_single_sinr = dict()
    for num_bits_per_symbol, code_rate, mcs_index in zip(modulation_order_list_clipped, code_rates_list_cliped, mcs_indicies_cliped):
        bler = single_MCS_simulation(num_bits_per_symbol, code_rate, default_sim_paras, ebno_db)
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
    ue_speed = 10
    delay_spread = 100e-9#
    sim_para_default = Simulation_Parameter(resouce_grid, batch_size, num_bits_per_symbol, code_rate, carrier_frequency, ue_speed, delay_spread)
    
    cqi_sinr = get_CQI()
    sinr_list = cqi_sinr["SINR"]
    cqi_code_list = cqi_sinr["cqi_code"]
    sinr_mcs_bler_dict = dict()
    for sinr, cqi_code in zip(sinr_list, cqi_code_list):
        print("NOw CQI {} is simulating".format(cqi_code))
        bler_list_single_sinr = single_SNR_simulation(sim_para_default, sinr)
        sinr_mcs_bler_dict[cqi_code] = bler_list_single_sinr


if __name__ == "__main__":
    main()







    






