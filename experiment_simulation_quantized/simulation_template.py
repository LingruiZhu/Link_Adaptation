import numpy as np
import h5py

import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

from la_agent.olla_dqn_agent import OLLA_DQN_agent
from la_agent.olla_agent import OuterLoopLinkAdaptation
from la_agent.olla_dqn_agent_simple import OLLA_DQN_agent_simple

from link_envinorment import LutEnvironment
from InterferenceToy import InterferenceToy
from quantization import UniformQuantizer


def generate_file_name(agent_name:str, num_trails:int, cqi_interval:int, num_quant_bits:int, cqi_delay:int, \
                    ack_weight:float, add_interference:bool, simulation_name:str, q_network_input_time_dimension:int):
    file_name = f"{agent_name}_ack_weight_{ack_weight}_num_time_slots_{q_network_input_time_dimension}_num_trails_{num_trails}_\
        cqi_interval_{cqi_interval}_quant_bits_{num_quant_bits}_delay_{cqi_delay}_add_interference_{add_interference}_{simulation_name}"
    return file_name


def olla_simulation_single_trail(olla_agent:OLLA_DQN_agent or OuterLoopLinkAdaptation or OLLA_DQN_agent_simple,
                                num_time_slots:int, cqi_intervals:int, cqi_delay:int=0, num_quant_bits:int=4, add_interference:bool=True):
    # TODO: Add delay part to the simulation
    # olla_dqn_agent = OLLA_DQN_agent(bler_target=0.1, 
    #                             data_file="/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
    #                             olla_step_size=0.05,
    #                             reliablity_weight=dqn_ack_weight)
    olla_agent.reset()
    
    lut_link_env = LutEnvironment(ebno_db=10.5)
    lut_link_env.set_interference_mode("time")
    interference1 = InterferenceToy(tx_power=0.1, h_factor=0.3, interval=10, period=2)
    interference2 = InterferenceToy(tx_power=0.1, h_factor=0.2, interval=6, period=3)
    if add_interference:
        lut_link_env.add_interference_noise(interference1)
        lut_link_env.add_interference_noise(interference2)

    sinr_quantizer = UniformQuantizer(min=6, max=12, num_bits=num_quant_bits)

    modu_order = 2
    code_rate = 602/1024

    # initialize result list
    ack_list = list()
    tbs_list = list()
    sinr_list = list()
    mcs_list = list()
    action_list = list()
    olla_sinr_list = list()
    real_sinr_list = list()
    sinr_offset_list = list()

    for time_index in range(num_time_slots):
        ack, tbs_size, _, eff_sinr = lut_link_env.step(modu_order, code_rate)
        real_sinr_list.append(eff_sinr)
        
        # collect data for training DQN
        if len(ack_list)>1:
            if olla_agent.__class__.__name__ == "OLLA_DQN_agent" or olla_agent.__class__.__name__  ==  "OLLA_DQN_agent_simple":
                olla_agent.add_memory(sinr=real_sinr_list[-2],
                              ack=ack_list[-1],
                              mcs_index=mcs_index,
                              action = action_index,
                              sinr_new=eff_sinr,
                              ack_new=ack)
                print("Now adding memory for agent")
                check_here = 1
            
        # update cqi every num_time slots
        if time_index % cqi_intervals == 0:
            if time_index - cqi_delay < 0:
                sinr_feedback = sinr_quantizer(real_sinr_list[0])
            else:
                sinr_feedback = sinr_quantizer(real_sinr_list[time_index-cqi_delay])
                
        if olla_agent.__class__.__name__ == "OLLA_DQN_agent":
            mcs_index, code_rate, modu_order, sinr_adjusted, action_index, sinr_offset = olla_agent.determine_mcs_action_from_sinr(sinr_feedback, ack)
        elif olla_agent.__class__.__name__ == "OLLA_DQN_agent_simple":
            mcs_index, code_rate, modu_order, sinr_adjusted, action_index, sinr_offset = olla_agent.determine_mcs_action_from_sinr(sinr_feedback, ack)
        else:
            mcs_index, code_rate, modu_order, sinr_adjusted, sinr_offset = olla_agent.determine_mcs_action_from_sinr(sinr_feedback, ack)
            action_index = 0
        # save result
        ack_list.append(ack)
        tbs_list.append(tbs_size)
        sinr_list.append(sinr_feedback)
        mcs_list.append(mcs_index)
        olla_sinr_list.append(sinr_adjusted)
        action_list.append(action_index)
        sinr_offset_list.append(sinr_offset)
        if olla_agent.__class__.__name__ == "OuterLoopLinkAdaptation":
            loss_history = None
        else:
            loss_history = olla_agent.loss_history
    return ack_list, tbs_list, sinr_list, mcs_list, olla_sinr_list, action_list, sinr_offset_list, loss_history, real_sinr_list


def OLLA_simulation(olla_agent, num_trails, num_time_slots, feedback_interval, num_quant_bits, cqi_delay:int, file_name:str, add_interference:bool):
    ack_result_list = list()
    tbs_result_list = list()
    sinr_feedback_result_list = list()
    mcs_result_list = list()
    sinr_result_list = list()
    action_result_list = list()
    sinr_offset_result_list = list()
    loss_history_result_list = list()
    real_sinr_result_list = list()

    for idx_train in range(num_trails):
        ack_list_single_trail, tbs_list_single_trail, sinr_feedback_list, mcs_list_single_trail, sinr_list_single_trail,\
            action_list_single_trail, sinr_offset_list_single_trail, loss_history_single_trail, real_sinr_list_single_trail = \
            olla_simulation_single_trail(olla_agent, num_time_slots, cqi_intervals=feedback_interval, num_quant_bits=num_quant_bits,\
                cqi_delay=cqi_delay, add_interference=add_interference)
        ack_result_list.append(ack_list_single_trail)
        tbs_result_list.append(tbs_list_single_trail)
        sinr_feedback_result_list.append(sinr_feedback_list)
        mcs_result_list.append(mcs_list_single_trail)
        sinr_result_list.append(sinr_list_single_trail)
        action_result_list.append(action_list_single_trail)
        sinr_offset_result_list.append(sinr_offset_list_single_trail)
        loss_history_result_list.append(loss_history_single_trail)
        real_sinr_result_list.append(real_sinr_list_single_trail)

    # convert type of data
    ack_result_array = np.array(ack_result_list)
    tbs_result_array = np.array(tbs_result_list)
    sinr_feedback_result_array = np.array(sinr_feedback_result_list)
    mcs_result_array = np.array(mcs_result_list)
    sinr_result_array = np.array(sinr_result_list)
    action_result_array = np.array(action_result_list)
    sinr_offset_result_array = np.array(sinr_offset_result_list)
    loss_history_result_array = np.array(loss_history_result_list)
    real_sinr_result_array = np.array(real_sinr_result_list)
    
    if olla_agent.__class__.__name__ == "OLLA_DQN_agent":
        file_path = "results_v1/" + file_name +".h5"
    elif olla_agent.__class__.__name__ == "OLLA_DQN_agent_simple" or olla_agent.__class__.__name__ == "OuterLoopLinkAdaptation":
        file_path = "results_v2/" + file_name +".h5"
    
    olla_lut_file = h5py.File(file_path, "w")
    olla_lut_file.create_dataset(name="ACK", data=ack_result_array)
    olla_lut_file.create_dataset(name="TBS", data=tbs_result_array)
    olla_lut_file.create_dataset(name="SINR_feedback", data=sinr_feedback_result_array)
    olla_lut_file.create_dataset(name="MCS", data=mcs_result_array)
    olla_lut_file.create_dataset(name="SINR", data=sinr_result_array)
    olla_lut_file.create_dataset(name="num_trails", data=num_trails)
    olla_lut_file.create_dataset(name="num_time_slots", data=num_time_slots)
    olla_lut_file.create_dataset(name="feedback_interval", data=feedback_interval)
    olla_lut_file.create_dataset(name="action_index", data=action_result_array)
    olla_lut_file.create_dataset(name="sinr_offset", data = sinr_offset_result_array)
    olla_lut_file.create_dataset(name="real_SINR", data=real_sinr_result_array)
    if loss_history_result_array[0].all() != None:
        olla_lut_file.create_dataset(name="loss_history", data=loss_history_result_array)
        check_here = 1
    olla_lut_file.close()


def la_simulation():
    # LA algorithms
    agent_name = "OLLA_DQN_v2"     # OLLA or OLLA_DQN_v1 or OLLA_DQN_v2
    ack_weight = 0
    simulation_name = "simulation_1213"
    
    # simulation tuning
    num_trails = 20
    num_time_slots = 1000
    input_time_dimension = 4
    
    # cqi configuration
    feedback_interval = 5
    num_quant_bits = 4
    cqi_delay = 0
    add_interference = True
    
    file_name = generate_file_name(agent_name, num_trails, feedback_interval, num_quant_bits, cqi_delay, ack_weight, add_interference,\
        simulation_name, input_time_dimension)
    
    olla_dqn_agent = OLLA_DQN_agent(bler_target=0.1, 
                                data_file="/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                olla_step_size=0.1,
                                reliablity_weight=ack_weight)
    olla_dqn_agent_simple = OLLA_DQN_agent_simple(bler_target=0.1,
                                data_file="/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                olla_step_size= 0.1,
                                reliablity_weight=ack_weight,
                                num_time_slots_q_network=input_time_dimension)
    olla_agent = OuterLoopLinkAdaptation(bler_target=0.1, 
                                data_file="/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                olla_step_size=0.1)
    
    if agent_name == "OLLA_DQN_v1": 
        simulation_agent = olla_dqn_agent        
    elif agent_name == "OLLA_DQN_v2":
        simulation_agent = olla_dqn_agent_simple
    elif agent_name == "OLLA":
        simulation_agent = olla_agent
    
    OLLA_simulation(simulation_agent, num_trails, num_time_slots, feedback_interval, num_quant_bits, cqi_delay=cqi_delay, file_name=file_name, add_interference=add_interference)


if __name__ == "__main__":
    la_simulation()
    