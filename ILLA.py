from multiprocessing import current_process
import numpy as np
from LinkSimulation import Link_Simulation

class ILLA:
    def __init__(self, awgn_cqi_bler_lut: dict, bler_target:float) -> None:
        """initialize function

        Args:
            awgn_cqi_bler_lut (dict): the look up table containing bler table for each cqi, the dimension is like (CQI, MCS, BLER)
        """
        # the following three for history
        self.cqi_history = list()
        self.mcs_history = list()
        self.tp_history = list()

        # look up table and bler target
        self.cqi_bler_lut = awgn_cqi_bler_lut
        self.bler_target = bler_target
        
        # 

    def step(self, cqi:int):
        current_cqi_lut = self.cqi_bler_lut[cqi]
        # TODO: here to add tbs block size.

        


if __name__ == "__main__":
    LUT_awgn = np.load("BLER_LUT_data/table3_LUT_AWGN_channel.npy", allow_pickle=True).item()
    print(LUT_awgn.keys())


