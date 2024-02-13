import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

import numpy as np
import random
import h5py
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from la_agent.olla_agent import OuterLoopLinkAdaptation
from la_agent.ExperienceBuffer import ExperienceBuffer


class Q_network(tf.keras.Model):
    def __init__(self, num_states:int, num_hidden_layers:list, num_actions:int):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten(input_shape=(1, num_states))
        self.hidden_layers = list()
        for size in num_hidden_layers:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation="relu"))
        self.output_layer = tf.keras.layers.Dense(num_actions)
    
    @tf.function 
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output


class OLLA_DQN_agent(OuterLoopLinkAdaptation):
    def __init__(self, bler_target, data_file, olla_step_size=0.1, reliablity_weight=0, num_time_slots:int=1, gamma:float=0.5) -> None:
        super().__init__(bler_target, data_file, olla_step_size)
        self.num_time_slots = num_time_slots
        self.state_size = 2*num_time_slots
        self.action_size = 5
        self.olla_offset_adjustments = [-0.5, -0.1, 0, 0.1, 0.5]
        self.epsilon = 1.0
        self.epsilon_decay = 0.9997 
        self.epsilon_min = 0.1
        self.gamma = gamma     # default = 0.95
        
        # initialize q networks (q_primary and q_target)
        num_layers = [128, 256, 128]
        self.q_target = Q_network(num_states=self.state_size, num_hidden_layers=num_layers, num_actions=self.action_size)
        self.q_primary = Q_network(num_states=self.state_size, num_hidden_layers=num_layers, num_actions=self.action_size)
        self.q_target.compile(loss="mse", optimizer=Adam())
        self.q_primary.compile(loss="mse", optimizer=Adam())
        self.q_network = self._build_q_network()

        self.ack_reward_weight = reliablity_weight
        self.batch_size = 128
        
        self.buffer_size = 5 *self.batch_size
        self.memory = ExperienceBuffer(self.buffer_size)
        self.state_old_list:list = list()
        self.state_new_list:list = list()
        self.is_state_ready = False
        
        self.sinr_offset_max = 3
        self.sinr_offset_min = -3
        
        self.mcs_history = list()
        self.action_index_history = list()
        self.loss_history = list()
        
        self.previous_sinr_offset = 0
    
    
    def reset(self):
        self.loss_history = list()
        self.memory = ExperienceBuffer(self.buffer_size)
        self.mcs_history = list()
        self.action_index_history = list()
        self.q_network = self._build_q_network()
        self.epsilon = 1.0
        self.state_old_list:list = list()
        self.state_new_list:list = list()
        self.is_state_ready = False
        self.previous_sinr_offset = 0
        
    
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
    
    
    def build_up_state(self, sinr, ack, sinr_new, ack_new):
        self.state_old_list.append([sinr, ack])
        self.state_new_list.append([sinr_new, ack_new])
        if len(self.state_old_list) >= self.num_time_slots:
            self.is_state_ready = True
            if len(self.state_old_list) > self.num_time_slots:
                self.state_old_list.pop(0)
                self.state_new_list.pop(0)
            self.state_old = np.array(self.state_old_list).flatten(order="c")
            self.state_new = np.array(self.state_new_list).flatten(order="c")
        check_here = 1

    
    def add_memory(self, sinr, ack, mcs_index, action, sinr_new, ack_new):
        self.build_up_state(sinr, ack, sinr_new, ack_new)
        reward = self.calculate_reward_value(ack_new, mcs_index)
        if self.is_state_ready:
            self.memory.add_experience(self.state_old, action, reward, self.state_new)
    
    
    def add_memory_only_sinr_offset(self, sinr, ack, mcs_index, action, sinr_new, ack_new):
        # this function need to be called every time slot, otherwise states cannot be built correctly
        self.build_up_state(self.previous_sinr_offset, ack, self.sinr_offset, ack_new)
        reward = self.calculate_reward_value(ack_new, mcs_index)
        if self.is_state_ready:
            self.memory.add_experience(self.state_old, action, reward, self.state_new)
        
    
    def calculate_reward_value(self, ack, mcs_index):
        tbs_normalized = self.trans_block_size_list_normalized[mcs_index]
        reward = tbs_normalized*ack + self.ack_reward_weight*ack
        return reward
    
    
    def update_q_network(self):
        print("now updating the q network . . .")
        # in the context of RL, this part is called replay
        minibatch = self.memory.sample(self.batch_size)
        input_q_network, output_q_network = [], []
        for state, action_index, reward, next_state in minibatch:
            next_state = np.expand_dims(next_state, axis=0)
            state = np.expand_dims(state, axis=0)
            target = reward + self.gamma*np.max(self.q_network.predict(next_state))     # take Q-value as the output (target) off-policy
            # target = reward                                                             # simplified
            targets_pool = self.q_network.predict(state)
            targets_pool[0][action_index] = target      # Add index 0 because of the dimension of model input/output is [batch_size, input/output size]
            input_q_network.append(state[0])
            output_q_network.append(targets_pool[0])
        history = self.q_network.fit(np.array(input_q_network), np.array(output_q_network), epochs=10, verbose=0)
        loss = history.history["loss"][0]
        return loss
    
    
    def update_q_network_primary(self):
        minibatch = self.memory.sample(self.batch_size)
        input_q_primary, output_q_primary = [], []
        for state, action_index, reward, next_state in minibatch:
            next_state = np.expand_dims(next_state, axis=0)
            state = np.expand_dims(state, axis=0)
            action_idx_primary = np.argmax(self.q_primary.predict(next_state))
            estimation_from_q_target = self.q_target.predict(next_state)[0][action_idx_primary]
            target = reward + self.gamma * estimation_from_q_target     # 
            # target = reward                                                             # simplified
            targets_pool = self.q_primary.predict(state)
            targets_pool[0][action_index] = target      # Add index 0 because of the dimension of model input/output is [batch_size, input/output size]
            input_q_primary.append(state[0])
            output_q_primary.append(targets_pool[0])
        history = self.q_network.fit(np.array(input_q_primary), np.array(output_q_primary), epochs=10, verbose=0)
        
        
    
    
    def save_q_network(self, save_path:str):
        self.q_network.save(save_path)
        
    
    def select_reasonable_actions(self):
        feasible_action_list = list()
        for i in range(self.action_size):
            new_sinr_offset = self.sinr_offset + self.olla_offset_adjustments[i]
            if new_sinr_offset <= self.sinr_offset_max and new_sinr_offset >= self.sinr_offset_min:
                feasible_action_list.append(i)
        return feasible_action_list
    
    
    def filter_out_nonreasonable_rewards(self, reward_list:list, action_index:list):
        min_element = min(reward_list)
        for i in range(len(reward_list)):
            if i not in action_index:
                reward_list[i] = min_element - 10
        return reward_list
           
    
    def update_olla_offset(self, ack: int, sinr:float):
        possible_actions_list = self.select_reasonable_actions()
        print(f"current sinr offset is {self.sinr_offset}")
        print(f"current possible action list is: {possible_actions_list}")
        
        if len(possible_actions_list) == 0:
            raise ValueError(f"Now possible action list is empty, current sinr offset is {self.sinr_offset}")
        
        if self.memory.get_buffer_length() < 2*self.batch_size:
            print("randomly_selecting")
            action_index = random.choice(possible_actions_list)
        else:
            self.update_epsilon()
            if np.random.rand() < self.epsilon:
                print("randomly_selecting")
                action_index = random.choice(possible_actions_list)
            else:
                print("according to sinr")
                state = np.expand_dims(self.state_new, axis=0)  
                action_values = self.q_network.predict(state)
                action_rewards_list = action_values[0]
                action_rewards_list = self.filter_out_nonreasonable_rewards(action_rewards_list, possible_actions_list)
                action_index = np.argmax(action_rewards_list)
        print(f"action index is {action_index}")
        print(f"olla offset adjustment is {self.olla_offset_adjustments[action_index]}")
        self.previous_sinr_offset = self.sinr_offset
        self.sinr_offset += self.olla_offset_adjustments[action_index]
        self.action_index_history.append(action_index)
        return action_index
    
    
    def determine_mcs_action_from_sinr(self, sinr_eff: float, ack: int):
        if self.memory.get_buffer_length() >= 5*self.batch_size:
            loss = self.update_q_network()
            self.loss_history.append(loss)
        action_index = self.update_olla_offset(ack, sinr_eff)
        effect_sinr = sinr_eff - self.sinr_offset
        sinr_idx = (np.abs(np.array(self.sinr_list) - effect_sinr)).argmin()
        successful_tbs_list = self.calculate_successful_tbs(sinr_idx)
        mcs_index, code_rate, num_bits_per_symbol = self.select_mcs(successful_tbs_list)
        self.mcs_history.append(mcs_index)
        return mcs_index, code_rate, num_bits_per_symbol, effect_sinr, action_index, self.sinr_offset
            
        