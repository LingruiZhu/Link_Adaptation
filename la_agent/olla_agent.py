import numpy as np
import h5py
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/zhu/Codes/link_adaptation")
sys.path.append("/home/zhu/Codes/link_abstractor")

from MCS_and_CQI import get_CQI, get_MCS
from LogisticLinkAbstractor import LogisticLinkAbstractor


class OuterLoopLinkAdaptation():
    # TODO: think about what kind of information OLLA needs actually?
    def __init__(self, bler_target, data_file, olla_step_size=0.1, show_bler_updates:bool=False) -> None:
        self.bler_target = bler_target
        self.sinr_offset = 0
        self.olla_step_size = olla_step_size
        self.step_size_down = self.olla_step_size
        self.step_size_up = ((1 - bler_target) / bler_target) * olla_step_size 
        
        self.cqi_table = get_CQI()

        result_data = h5py.File(data_file, "r")
        self.sinr_list = result_data["sinr_list"]
        self.sinr_mcs_bler = result_data["sinr_mcs_bler_array"]

        mcs_table = get_MCS()
        self.mcs_index_list = mcs_table.mcs_index
        self.mcs_code_rate = mcs_table.code_rate
        self.mcs_number_bits_per_symbol = mcs_table.number_bits_per_symbol
        self.trans_block_size_list = mcs_table.tbs
        self.trans_block_size_list_normalized = self.trans_block_size_list / np.mean(self.trans_block_size_list)
        
        self.show_bler_updates = show_bler_updates
        
        self.link_abstractor = LogisticLinkAbstractor(parameter_file="/home/zhu/Codes/link_abstractor/bler_logistic_parameter.h5")
    
    
    def reset(self):
        self.sinr_offset = 0


    def plot_reward_normalized(self):
        plot_tbs = self.trans_block_size_list
        plot_tbs.insert(0, 0)
        plot_tbs_max_norm = plot_tbs / np.max(plot_tbs)
        plot_tbs_mean_norm = plot_tbs / np.mean(plot_tbs)
        plt.figure()
        plt.plot(plot_tbs, "r-x", label="No Normalization")
        plt.plot(plot_tbs_mean_norm,  "b--d", label="Mean Normalization")
        plt.plot(plot_tbs_max_norm, "g-.o", label="Max Normalization")
        plt.legend()
        plt.xticks(np.arange(0, len(plot_tbs)))
        plt.xlabel("MCS index")
        plt.ylabel("Reward")
        plt.grid()
        
        plt.figure()
        plt.plot(plot_tbs_mean_norm, "b--d", label="Mean Normalization")
        plt.plot(plot_tbs_max_norm, "g-.o", label="Max Normalization")
        plt.xlabel("MCS index")
        plt.ylabel("Reward")
        plt.xticks(np.arange(0, len(plot_tbs)))
        plt.legend()
        plt.grid()
        
        plt.show()
        

    def update_agent(self, ack:int):
        """update SINR offset accroding to ACK/NACK

        Args:
            ack (int): 1 means ACk and 0 means NACK
        """
        if ack == 0:
            if self.show_bler_updates:
                print("Now ack = 0")
            self.sinr_offset += self.step_size_up
        else:
            if self.show_bler_updates:
                print("Now ack = 1")
            self.sinr_offset -= self.step_size_down
        if self.show_bler_updates:
            print(f"SINR offset has been adjusted: {self.sinr_offset}")
    
    
    def calculate_successful_tbs(self, sinr_idx:int, eff_sinr:float):
        # mcs_bler_list = list(self.sinr_mcs_bler[sinr_idx,:])
        self.mcs_bler_list = self.link_abstractor.calculate_bler_mcs_list(eff_sinr)
        reward_list = list()
        for bler, tbs in zip(self.mcs_bler_list, self.trans_block_size_list):
            if bler <= self.bler_target:
                reward = (1-bler)*tbs
            else:
                reward = 0
            reward_list.append(reward)
        return reward_list
    

    def select_mcs(self, reward_list):
        mcs_index = reward_list.index(max(reward_list))
        code_rate = self.mcs_code_rate[mcs_index]
        num_bits_per_symbol = self.mcs_number_bits_per_symbol[mcs_index]
        bler_estimate = self.mcs_bler_list[mcs_index]
        return mcs_index, code_rate, num_bits_per_symbol, bler_estimate

        
    def determine_mcs_action(self, cqi:int, ack:int):
        """determining mcs based on CQI

        Args:
            cqi (int): _description_
        """
        self.update_agent(ack)
        sinr_from_cqi = self.cqi_table.estimate_sinr_from_cqi(cqi)
        effect_sinr = sinr_from_cqi - self.sinr_offset
        sinr_idx = (np.abs(np.array(self.sinr_list) - effect_sinr)).argmin()
        successful_tbs_list = self.calculate_successful_tbs(sinr_idx, effect_sinr)
        mcs_index, code_rate, num_bits_per_symbol, bler_estimate = self.select_mcs(successful_tbs_list)
        return mcs_index, code_rate, num_bits_per_symbol, effect_sinr
    

    def determine_mcs_action_from_sinr(self, sinr_eff:float, ack:int):
        self.update_agent(ack)
        effect_sinr = sinr_eff - self.sinr_offset
        sinr_idx = (np.abs(np.array(self.sinr_list) - effect_sinr)).argmin()
        successful_tbs_list = self.calculate_successful_tbs(sinr_idx, effect_sinr)
        mcs_index, code_rate, num_bits_per_symbol, bler_estimate = self.select_mcs(successful_tbs_list)
        return mcs_index, code_rate, num_bits_per_symbol, effect_sinr, self.sinr_offset, bler_estimate


if __name__ == "__main__":
    pass



        
        