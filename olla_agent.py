import numpy as np
import h5py

from MCS_and_CQI import get_CQI, get_MCS

class OuterLoopLinkAdaptation():
    # TODO: think about what kind of information OLLA needs actually?
    def __init__(self, bler_target, data_file, olla_step_size=0.1) -> None:
        self.bler_target = bler_target
        self.sinr_offset = 0
        self.olla_step_size = olla_step_size
        self.step_size_down = self.olla_step_size
        self.step_size_up = (1 - bler_target) / bler_target
        
        self.cqi_table = get_CQI()

        result_data = h5py.File(data_file, "r")
        self.sinr_list = result_data["sinr_list"]
        self.sinr_mcs_bler = result_data["sinr_mcs_bler_array"]

        mcs_table = get_MCS()
        self.mcs_index_list = mcs_table.mcs_index
        self.mcs_code_rate = mcs_table.code_rate
        self.mcs_number_bits_per_symbol = mcs_table.number_bits_per_symbol
        self.trans_block_size_list = mcs_table.tbs


    def update_agent(self, ack:int):
        """update SINR offset accroding to ACK/NACK

        Args:
            ack (int): 1 means ACk and 0 means NACK
        """
        if ack == 1:
            self.sinr_offset += self.olla_step_size
        else:
            self.sinr_offset -= self.olla_step_size / (1 - self.bler_target) * self.bler_target
    

    def calculate_reward(self, sinr_idx:int):
        mcs_bler_list = list(self.sinr_mcs_bler[sinr_idx,:])
        reward_list = list()
        for bler, tbs in zip(mcs_bler_list, self.trans_block_size_list):
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
        return mcs_index, code_rate, num_bits_per_symbol

        
    def determine_mcs_action(self, cqi:int, ack:int):
        """determining mcs based on CQI

        Args:
            cqi (int): _description_
        """
        self.update_agent(ack)
        sinr_from_cqi = self.cqi_table.estimate_sinr_from_cqi(cqi)
        effect_sinr = sinr_from_cqi + self.sinr_offset
        sinr_idx = (np.abs(np.array(self.sinr_list) - effect_sinr)).argmin()
        reward_list = self.calculate_reward(sinr_idx)
        mcs_index, code_rate, num_bits_per_symbol = self.select_mcs(reward_list)
        return mcs_index, code_rate, num_bits_per_symbol, effect_sinr


if __name__ == "__main__":
    pass



        
        