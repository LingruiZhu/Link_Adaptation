import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

from la_agent.olla_agent import OuterLoopLinkAdaptation


class NonLinear_OLLA(OuterLoopLinkAdaptation):
    def __init__(self, bler_target, data_file, olla_step_size=0.1, max_momentum=5, alpha_down=0.85, alpha_up=1) -> None:
        super().__init__(bler_target, data_file, olla_step_size)
        self.up_step_decay_factor = alpha_up
        self.down_step_decay_factor = alpha_down
        self.momentum_index = 0
        self.max_momentum = max_momentum
        self.preivous_ack = 0
        
        self.step_size_down = olla_step_size
        self.step_size_down_avg = olla_step_size*(self.down_step_decay_factor**self.max_momentum)
        self.step_size_up = ((1 - bler_target) / bler_target) * self.step_size_down
        
    
    def update_momentum_index(self, ack):
        if self.preivous_ack == ack:
            self.momentum_index += 1
        else:
            self.momentum_index = 0
            self.momentum_index += 1
        if self.momentum_index > self.max_momentum:
            self.momentum_index = self.max_momentum
        self.preivous_ack = ack
    
    
    def update_agent(self, ack):
        """update SINR offset accroding to ACK/NACK

        Args:
            ack (int): 1 means ACk and 0 means NACK
        """
        self.update_momentum_index(ack)
        if ack == 0:
            # print("Now ack = 0")
            self.sinr_offset += self.step_size_up*(self.up_step_decay_factor**self.momentum_index)
        else:
            # print("Now ack = 1")
            self.sinr_offset -= self.step_size_down*(self.down_step_decay_factor**self.momentum_index)
        # print(f"SINR offset has been adjusted: {self.sinr_offset}")
        # print("here update it nonlinearly")
        checkhere = 1


def Nolla_parameter():
    pass
    
        