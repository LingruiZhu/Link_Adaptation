import numpy as np
import gym


class LinkEnvironment:
    def __init__(self, ue_speed:float, cqi_interval:int, data_file:str) -> None:
        self.cqi_interval = cqi_interval
        self.ue_speed = ue_speed

        self.channel = None         # define in set_channel

        self.cqi = None             # state variables
        self.subframe_index = None  # state variables
        self.__get_offline_data(data_file)  # get mcs data
    
    def __get_offline_data(self, data_file):
        self.offline_data = np.load(data_file, allow_pickle=True)[()]
        self.mcs = {"modulation_order": self.offline_data["modulation_order"],
                    "transport_block_size": self.offline_data["block_size"]}        # Action space
        
    def step(self, mcs_index):
        modulation_order = self.mcs[modulation_order][mcs_index]
        block_size = self.mcs["transport_block_size"][mcs_index]
        

