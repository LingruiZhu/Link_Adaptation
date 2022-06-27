import numpy as np
import CQI_functions as cf

class OuterLoopLinkAdaptation():
    def __init__(self, bler_target, olla_window_size, awgn_data, olla_step_size=0) -> None:
        self.bler_target = bler_target
        self.sinr_offset = 0
        self.awgn_data = awgn_data
        self.olla_step_size = olla_step_size

        self.num_cqi = awgn_data["snr_vs_bler"].shape[1]
        self.cqi_to_estimated_snr = [cf.estimate_sinr_from_cqi(self.awgn_data, i) for i in range(self.num_cqi)]

    def update_agent(self, ack:int):
        """update SINR offset accroding to ACK/NACK

        Args:
            ack (int): 1 means ACk and 0 means NACK
        """
        if ack == 1:
            self.sinr_offset -= self.olla_step_size
        else:
            self.sinr_offset += self.olla_step_size*(1 - self.bler_target)/self.bler_target
        
    def determine_mcs_action(self, cqi:int):
        """determing mcs based on CQI

        Args:
            cqi (int): _description_
        """
        if cqi == 0:
            mcs = 0
        else:
            estimated_sinr = self.cqi_to_estimated_snr[cqi]
            adjusted_sinr = estimated_sinr + self.sinr_offset
            mcs = cf.determine_mcs_from_sinr(self.awgn, adjusted_sinr, self.bler_target)
        return mcs

        
        