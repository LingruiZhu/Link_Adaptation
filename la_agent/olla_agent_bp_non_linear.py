import sys
sys.path.append("/home/zhu/Codes/link_adaptation")
import numpy as np

from la_agent.olla_agent import OuterLoopLinkAdaptation


class NonLinear_OLLA_bp(OuterLoopLinkAdaptation):
    def __init__(self, bler_target, data_file, olla_step_size=0.1) -> None:
        super().__init__(bler_target, data_file, olla_step_size)
        self.learning_rate_up = 0.001
        self.learning_rate_down = 0.001
        self.mcs_index_for_update = None
        self.bler_estimate_for_update = None
        self.down_step_decay_factor = 1
        self.up_step_decay_factor = 1
    
    
    def update_decay_factor(self, ack):
        real_mcs_index = self.mcs_index_for_update + 9
        beta_1 = self.link_abstractor.parameter_dictionary[str(real_mcs_index)][-1][0]
        if ack:
            self.down_step_decay_factor -= self.learning_rate_down * self.step_size_down * self.bler_estimate_for_update * beta_1
            print("have updated down decay factor: " + str(self.down_step_decay_factor))
        else: 
            self.up_step_decay_factor -= self.learning_rate_up * self.step_size_up * (1 - self.bler_estimate_for_update) * beta_1
            if self.up_step_decay_factor <=0.95:
                self.up_step_decay_factor = 0.95
            print("have updated up decay factor: " + str(self.up_step_decay_factor))
    
    
    def determine_mcs_action_from_sinr(self, sinr_eff: float, ack: int):
        if self.mcs_index_for_update is not None and self.bler_estimate_for_update is not None:
            self.update_decay_factor(ack)
        mcs_index, code_rate, num_bits_per_symbol, effect_sinr, sinr_offset, bler_estimate = super().determine_mcs_action_from_sinr(sinr_eff, ack)       # TODO: get output from super().function and give out in the current function.
        self.mcs_index_for_update = mcs_index
        if isinstance(bler_estimate, np.ndarray):
            self.bler_estimate_for_update = bler_estimate[0]
        else:
            self.bler_estimate_for_update = bler_estimate
        return mcs_index, code_rate, num_bits_per_symbol, effect_sinr, self.sinr_offset, bler_estimate
    
    
    def update_agent(self, ack: int):
        if ack == 0:
            if self.show_bler_updates:
                print("Now ack = 0")
            self.sinr_offset += self.up_step_decay_factor * self.step_size_up
        else:
            if self.show_bler_updates:
                print("Now ack = 1")
            self.sinr_offset -= self.down_step_decay_factor * self.step_size_down
        print(f"SINR offset has been adjusted: {self.sinr_offset}")
        if self.show_bler_updates:
            print(f"SINR offset has been adjusted: {self.sinr_offset}")
        