import numpy as np
import h5py
import random

import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

from MCS_and_CQI import get_CQI, get_MCS
from la_agent.olla_dqn_agent import OLLA_DQN_agent


class OLLA_DQN_agent_simple(OLLA_DQN_agent):
    def __init__(self, bler_target, data_file, olla_step_size=0.1, reliablity_weight=0, num_time_slots_q_network:int=0) -> None:
        super().__init__(bler_target, data_file, olla_step_size, reliablity_weight)
        self.olla_offset_choices = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        self.action_size = len(self.olla_offset_choices)
        self.state_size = (num_time_slots_q_network) * 2
        self.num_time_slots_q_network = num_time_slots_q_network
        self.stack_length = num_time_slots_q_network
        self.sinr_list
        self.ack_stack = list()
        self.sinr_stack = list()
        self.ack_stack_for_predict = list()         # here to initialize them and then 
        self.sinr_stack_for_predict = list()
        

    def reset(self):
        super().reset()
        self.ack_stack = list()
        self.sinr_stack = list()
        self.ack_stack_for_predict = list()         
        self.sinr_stack_for_predict = list()
    
    
    def add_new_stack_element(self, stack_list:list, new_element:any):
        stack_list.append(new_element)
        if len(stack_list) > self.stack_length:
            stack_list.pop(0)
        
    
    def update_olla_offset(self, ack: int, sinr: float):
        self.add_new_stack_element(self.ack_stack_for_predict, ack)
        self.add_new_stack_element(self.sinr_stack_for_predict, sinr)
        
        state = np.array(self.sinr_stack_for_predict+self.ack_stack_for_predict)
        state = np.expand_dims(state, axis=0)
        self.update_epsilon()
        if np.random.rand() < self.epsilon or len(self.ack_stack_for_predict) < 2*self.num_time_slots_q_network: 
            action_index = random.randrange(self.action_size)
        else:
            action_values = self.q_network.predict(state)
            action_index = np.argmax([action_values[0]])
        self.sinr_offset = self.olla_offset_choices[action_index]
        self.action_index_history.append(action_index)
        return action_index


    def add_memory(self, sinr, ack, mcs_index, action, sinr_new, ack_new):
        self.sinr_stack.append(sinr)
        self.ack_stack.append(ack)
        if len(self.ack_stack) >= self.num_time_slots_q_network:
            if len(self.ack_stack) > self.num_time_slots_q_network:
                self.ack_stack.pop(0)
                self.sinr_stack.pop(0)
            old_state = np.array(self.sinr_stack+self.ack_stack)
            new_state = None
            reward = self.calculate_reward_value(ack_new, mcs_index)
            self.memory.append((old_state, action, reward, new_state))
            check_here = 1
        print("Here is Son adding memory")
    
                        
        