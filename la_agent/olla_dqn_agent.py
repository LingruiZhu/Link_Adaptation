import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

import numpy as np
import random
import h5py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from la_agent.olla_agent import OuterLoopLinkAdaptation


class OLLA_DQN_agent(OuterLoopLinkAdaptation):
    def __init__(self, bler_target, data_file, olla_step_size=0.1, reliablity_weight=5) -> None:
        super().__init__(bler_target, data_file, olla_step_size)
        self.state_size = 2
        self.action_size = 3
        self.olla_offset_adjustments = [-1, 0, +1]
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1
        self.gamma = 0.95
        self.q_network = self._build_q_network()
        self.memory = list()
        self.ack_reward_weight = reliablity_weight
        self.batch_size = 20
        
        self.mcs_history = list()
        self.action_index_history = list()
        self.loss_history = list()
    
    
    def reset(self):
        self.loss_history = list()
        self.memory = list()
        self.mcs_history = list()
        self.action_index_history = list()
        self.q_network = self._build_q_network()
        self.epsilon = 1.0
        
    
    def _build_q_network(self):
        q_network = Sequential()
        q_network.add(Dense(16, input_shape=(self.state_size,), activation="relu"))
        q_network.add(Dense(32, activation="relu"))
        q_network.add(Dense(self.action_size, activation="linear"))
        q_network.compile(loss="mse", optimizer=Adam())
        return q_network
    
    
    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
    
    
    def add_memory(self, sinr, ack, mcs_index, action, sinr_new, ack_new):
        old_state = np.array([sinr, ack])
        new_state = np.array([sinr_new, ack_new])
        reward = self.calculate_reward_value(ack_new, mcs_index)
        self.memory.append((old_state, action, reward, new_state))
        print("Here is DAD adding memory")
        print("now {} time samples have been collected".format(len(self.memory)))
        
    
    def calculate_reward_value(self, ack, mcs_index):
        tbs_size = self.trans_block_size_list[mcs_index]
        reward = tbs_size*ack + self.ack_reward_weight*ack
        return reward
    
    
    def update_q_network(self):
        print("now updating the q network . . .")
        # in the context of RL, this part is called replay
        minibatch = random.sample(self.memory, self.batch_size)
        input_q_network, output_q_network = [], []
        for state, action_index, reward, next_state in minibatch:
            next_state = np.expand_dims(next_state, axis=0)
            state = np.expand_dims(state, axis=0)
            # target = reward + self.gamma*np.max(self.q_network.predict(next_state))     # take Q-value as the output (target)
            target = reward                                                             # simuplified
            targets_pool = self.q_network.predict(state)
            targets_pool[0][action_index] = target      # Add index 0 because of the dimension of model input/output is [batch_size, input/output size]
            input_q_network.append(state[0])
            output_q_network.append(targets_pool[0])
        history = self.q_network.fit(np.array(input_q_network), np.array(output_q_network), epochs=10, verbose=0)
        loss = history.history["loss"][0]
        return loss
        
    
    def update_olla_offset(self, ack: int, sinr:float):
        state = np.array([sinr, ack])
        state = np.expand_dims(state, axis=0)
        self.update_epsilon()
        if np.random.rand() < self.epsilon:
            print("randomly_selecting")
            action_index = random.randrange(self.action_size)
        else:
            print("according to sinr")
            action_values = self.q_network.predict(state)
            action_index = np.argmax([action_values[0]])
        self.sinr_offset += self.olla_offset_adjustments[action_index]
        self.action_index_history.append(action_index)
        return action_index
    
    
    def determine_mcs_action_from_sinr(self, sinr_eff: float, ack: int):
        if len(self.memory) >= 2*self.batch_size:
            loss = self.update_q_network()
            self.loss_history.append(loss)
        action_index = self.update_olla_offset(ack, sinr_eff)
        effect_sinr = sinr_eff - self.sinr_offset
        sinr_idx = (np.abs(np.array(self.sinr_list) - effect_sinr)).argmin()
        successful_tbs_list = self.calculate_successful_tbs(sinr_idx)
        mcs_index, code_rate, num_bits_per_symbol = self.select_mcs(successful_tbs_list)
        self.mcs_history.append(mcs_index)
        return mcs_index, code_rate, num_bits_per_symbol, effect_sinr, action_index, self.sinr_offset
            
        