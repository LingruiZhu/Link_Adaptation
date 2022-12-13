import numpy as np
import h5py

import sys
sys.path.append("/home/zhu/Codes/link_adaptation")


from MCS_and_CQI import get_CQI, get_MCS


class ThompsonLinkAdaptation:
    def __init__(self, bler_target:float) -> None:
        self.bler_target = bler_target
        self.cqi_sinr = get_CQI()       # need to be further investigated for toy example size(mcs and cqi) 
        self.mcs = get_MCS()     
        self.num_sinr = len(self.cqi_sinr.sinr_list)
        self.num_mcs = len(self.mcs.code_rate)

        self.alpha_set = np.ones((self.num_sinr, self.num_mcs))         # for counting ACK
        self.beta_set = np.ones((self.num_sinr, self.num_mcs))          # for counting NACK

        self.previous_sinr_index = None
        self.previous_mcs_index = None
    

    def decide_sinr_index(self, sinr:float):
        sinr_idx = (np.abs(np.array(self.sinr_list) - sinr)).argmin()
        return sinr_idx
    

    def calculate_reward(self, sinr_index):
        "reward here is not based on beta distribution instead of having the LUT"
        bler_current_sinr = list(np.random.beta(self.alpha_set[sinr_index, :], self.beta_set[sinr_index,:]))
        reward_list = list()
        for bler, tbs in zip(bler_current_sinr, self.mcs.tbs):
            reward = bler * tbs
            reward_list.append(reward)
        return reward_list
    

    def select_mcs_from_reward(self, reward_list:list):
        mcs_index = reward_list.index(max(reward_list))
        code_rate = self.mcs.code_rate[mcs_index]
        num_bits_per_symbol = self.mcs.number_bits_per_symbol[mcs_index]
        return mcs_index, code_rate, num_bits_per_symbol


    def update(self, sinr_index:int, mcs_index:int, ack:bool):
        mcs_index_in_array = self.mcs.mcs_index.index(mcs_index)
        if ack:
            self.alpha_set[sinr_index, mcs_index_in_array] += 1
        else:
            self.beta_set[sinr_index, mcs_index_in_array] += 1


    def act(self, sinr:float, ack:bool):
        if self.previous_mcs_index is not None:
            self.update(self.previous_sinr_index, self.previous_mcs_index, ack)
        sinr_index = self.decide_sinr_index(sinr)
        reward_list = self.calculate_reward(sinr_index)
        mcs_index, code_rate, num_bits_per_symbol = self.select_mcs_from_reward(reward_list)
        self.previous_sinr_index = sinr_index
        self.previous_mcs_index = mcs_index 
        return mcs_index, code_rate, num_bits_per_symbol


if __name__ == "__main__":
    list_a = list(np.ones((5)))
    print(list_a)
