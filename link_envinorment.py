import numpy as np
import gym

from LinkSimulation import Link_Simulation
from LUTLinkSimulation import LUT_Link_simulation
from Simulation_Parameters import Simulation_Parameter, get_default_parameters, Channel_Model
from MCS_and_CQI import get_MCS
from InterferenceToy import InterferenceToy

class LinkEnvironment:
    def __init__(self, ebno_db:float=10) -> None:
        # define link simulation 
        default_sim_paras = get_default_parameters()
        default_sim_paras.channel_model = Channel_Model.CDL
        self.link_simulation = Link_Simulation(default_sim_paras)
        self.mcs_set = get_MCS()
        self.ebno_db = ebno_db

        
    def step(self, number_bits_per_symbol, code_rate):
        self.link_simulation.update_mcs(num_bits_per_symbol=number_bits_per_symbol, code_rate=code_rate)
        ack, tsb_size, cqi = self.link_simulation.simulate_single_PRB_random_channel(ebno_db=self.ebno_db)
        return ack, tsb_size, cqi
    

class LutEnvironment:
    def __init__(self, ebno_db:float=10) -> None:
        # define lut link simulation
                # define link simulation 
        default_sim_paras = get_default_parameters()
        default_sim_paras.channel_model = Channel_Model.CDL
        lut_file = "/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5"
        self.lut_simulation = LUT_Link_simulation(lut_file=lut_file, sim_paras=default_sim_paras)
        self.mcs_set = get_MCS()
        self.ebno_db = ebno_db
        self.interference_mode = None
    

    def add_interference_noise(self, interference_source:InterferenceToy):
        if self.interference_mode == "time":
            self.lut_simulation.add_interference_time(interference_source)
        elif self.interference_mode == "space":
            self.lut_simulation.add_interference_space(interference_source)
        else:
            raise ValueError("In the curretn interference mode, interference source cannot be added.")


    def set_interference_mode(self, mode:str):
        if mode in ["time", "space"]:
            self.interference_mode = mode
        else:
            raise ValueError("Please set mode as time or space")


    def step(self, num_bits_per_symbol, code_rate):
        if self.interference_mode == "space":
            ack, tbs, cqi, eff_sinr, position = self.lut_simulation.simulate_block_transmission(num_bits_per_symbol, code_rate, self.ebno_db, self.interference_mode)    
            return ack, tbs, cqi, eff_sinr, position
        else:
            ack, tbs, cqi, eff_sinr = self.lut_simulation.simulate_block_transmission(num_bits_per_symbol, code_rate, self.ebno_db, self.interference_mode)
            return ack, tbs, cqi, eff_sinr
    

    # def step_perfect_sinr_feedback(self, num_bits_per_symbol, code_rate):
    #     ack, tbs, _, eff_sinr = self.lut_simulation.simulate_block_transmission(num_bits_per_symbol, code_rate, self.ebno_db, self.add_interference)
    #     return ack, tbs, eff_sinr
    
    
if __name__ == "__main__":
    link_env = LinkEnvironment()
    link_env.step(modulation_order=2, code_rate=308/1024)
    

        

        

