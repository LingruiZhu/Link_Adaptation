import numpy as np
import h5py
import math

import sionna as sn
from Simulation_Parameters import Simulation_Parameter, Channel_Model
from sionna.channel import AWGN, GenerateOFDMChannel

from MCS_and_CQI import get_CQI, get_MCS


class LUT_Link_simulation:
    def __init__(self, lut_file:str, sim_paras:Simulation_Parameter) -> None:
        self.resource_grid = sim_paras.resource_grid
        self.carrier_frequency = sim_paras.carrier_frequency
        self.ue_speed = sim_paras.ue_speed
        self.delay_spread = sim_paras.delay_spread
        UE_Array = sn.channel.tr38901.Antenna( polarization="single",
                                            polarization_type="V",
                                            antenna_pattern="38.901",
                                            carrier_frequency=self.carrier_frequency)
        BS_Array = sn.channel.tr38901.AntennaArray(num_rows=1,
                                            num_cols=1,
                                            polarization="single",
                                            polarization_type="V",
                                            antenna_pattern="38.901", # Try 'omni'
                                            carrier_frequency=self.carrier_frequency)
        direction = "downlink"
        CDL_model = "C"
        CDL = sn.channel.tr38901.CDL(CDL_model,
                                    delay_spread=self.delay_spread,
                                    carrier_frequency=self.carrier_frequency,
                                    ut_array=UE_Array,
                                    bs_array=BS_Array,
                                    direction=direction,
                                    min_speed=self.ue_speed,
                                    max_speed=self.ue_speed)
        self.channel_type = sim_paras.channel_model
        if self.channel_type == Channel_Model.CDL:
            self.channel_model = CDL
            self.channel = sn.channel.OFDMChannel(CDL, self.resource_grid, add_awgn=True, normalize_channel=True, return_channel=True)
        elif self.channel_type == Channel_Model.AWGN:
            self.channel = AWGN()
        self.resouce_grid = sim_paras.resource_grid
        
        # initiallize the look-up table for simulation
        self.lut_file = lut_file
        self.__initialize_lut()
        
        # initialize the modulation and coding schemes 
        mcs_table = get_MCS()
        self.mcs_list = list()
        for num_bits_symbol_element, code_rate_element in zip(mcs_table.number_bits_per_symbol, mcs_table.code_rate):
            self.mcs_list.append((num_bits_symbol_element, code_rate_element))
        self.tbs_list = mcs_table.tbs

        self.cqi_table = get_CQI()
    

    def __initialize_lut(self):
        # TODO: take all information from h5 file: including sinr cqi bler mcs. The core part is always the table
        dataset_file = h5py.File(self.lut_file, "r")
        self.sinr_list = list(dataset_file.get("sinr_list"))
        self.bler_array = np.array(dataset_file.get("sinr_mcs_bler_array"))
        dataset_file.close
    

    def estimate_bler_from_sinr(self, sinr, sinr_bler_column):
        sinr_index = (np.abs(np.array(self.sinr_list) - sinr)).argmin()
        if sinr_index == 0 or sinr_index == len(self.sinr_list)-1:
            estimated_bler = sinr_bler_column[sinr_index]
        else:
            if self.sinr_list[sinr_index] <= sinr:
                left_sinr = self.sinr_list[sinr_index]
                left_bler = sinr_bler_column[sinr_index]
                right_sinr = self.sinr_list[sinr_index+1]
                right_bler = sinr_bler_column[sinr_index+1]
            else:
                left_sinr = self.sinr_list[sinr_index-1]
                left_bler = sinr_bler_column[sinr_index-1]
                right_sinr = self.sinr_list[sinr_index]
                right_bler = sinr_bler_column[sinr_index]
            rate = (right_bler - left_bler) / (right_sinr - left_sinr)
            estimated_bler = left_bler + rate * (sinr - left_sinr)
        return estimated_bler


    def simulate_block_transmission(self, num_bits_per_symbol:int, code_rate:float, ebno_dB:float):
        # calculate the channel factor
        genereate_h = GenerateOFDMChannel(self.channel_model, self.resouce_grid)
        h_freq = genereate_h(batch_size=1)
        ebno_linear = 10**(ebno_dB/10)
        eff_sinr = np.mean(np.absolute(h_freq)**2) * ebno_linear        # accurately here should be efficient SINR
        eff_sinr_db = 10*math.log10(eff_sinr)
        cqi = self.cqi_table.decide_cqi_from_sinr(eff_sinr_db)

        # find the msc column index
        mcs_tuple = (num_bits_per_symbol, code_rate)
        mcs_index = self.mcs_list.index(mcs_tuple)
        sinr_bler_column = self.bler_array[:, mcs_index]

        # find the sinr index near the sinr
        estimated_bler = self.estimate_bler_from_sinr(eff_sinr_db, sinr_bler_column)
        rnd_number = np.random.uniform()
        if rnd_number < estimated_bler:
            ack = 0     # transmission falied
        else:
            ack = 1     # successful transmission
        tbs = self.tbs_list[mcs_index]

        return ack, tbs, cqi
