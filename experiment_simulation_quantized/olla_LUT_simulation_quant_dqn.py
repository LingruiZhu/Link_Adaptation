import numpy as np
import h5py

import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

from la_agent.olla_dqn_agent import OLLA_DQN_agent
from link_envinorment import LutEnvironment
from InterferenceToy import InterferenceToy
from quantization import UniformQuantizer


def olla_lut_simulation_perfect_dqn_single_trail(num_time_slots:int, cqi_intervals:int, num_quant_bits:int, dqn_ack_weight:float):
    # TODO: insert CQI intervals to simulation
    # initialize objects
    olla_dqn_agent = OLLA_DQN_agent(bler_target=0.1, 
                                data_file="/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                olla_step_size=0.05,
                                reliablity_weight=dqn_ack_weight)
    lut_link_env = LutEnvironment(ebno_db=10.5)
    lut_link_env.set_interference_mode("time")
    interference1 = InterferenceToy(tx_power=0.1, h_factor=0.3, interval=10, period=2)
    interference2 = InterferenceToy(tx_power=0.1, h_factor=0.2, interval=6, period=3)
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
    sinr_eff_list = list()
    env_sinr_list = list()

    for time_index in range(num_time_slots):
        ack, tbs_size, _, eff_sinr = lut_link_env.step(modu_order, code_rate)
        
        # collect data for training DQN
        if len(ack_list)>0:
            olla_dqn_agent.add_memory(sinr=env_sinr_list[-1],
                              ack=ack_list[-1],
                              mcs_index=mcs_index,
                              action = action_index,
                              sinr_new=eff_sinr,
                              ack_new=ack)
        
        # update cqi every num_time slots
        if time_index % cqi_intervals == 0:
            sinr_feedback = sinr_quantizer(eff_sinr)
        mcs_index, code_rate, modu_order, sinr_adjusted, action_index = olla_dqn_agent.determine_mcs_action_from_sinr(sinr_feedback, ack)

        # save result
        ack_list.append(ack)
        tbs_list.append(tbs_size)
        sinr_list.append(sinr_feedback)
        mcs_list.append(mcs_index)
        sinr_eff_list.append(sinr_adjusted)
        env_sinr_list.append(eff_sinr)
        action_list.append(action_index)
        
    return ack_list, tbs_list, sinr_list, mcs_list, sinr_eff_list, action_list


def dqn_olla_LUT_simulation_quant_sinr_feedback(num_trails, num_time_slots, feedback_interval, num_quant_bits, dqn_ack_weight):
    ack_result_list = list()
    tbs_result_list = list()
    sinr_feedback_result_list = list()
    mcs_result_list = list()
    sinr_result_list = list()
    action_result_list = list()

    for idx_train in range(num_trails):
        ack_list_single_trail, tbs_list_single_trail, sinr_feedback_list, mcs_list_single_trail, sinr_list_single_trail, action_list_single_trail = \
            olla_lut_simulation_perfect_dqn_single_trail(num_time_slots, feedback_interval, num_quant_bits, dqn_ack_weight)
        ack_result_list.append(ack_list_single_trail)
        tbs_result_list.append(tbs_list_single_trail)
        sinr_feedback_result_list.append(sinr_feedback_list)
        mcs_result_list.append(mcs_list_single_trail)
        sinr_result_list.append(sinr_list_single_trail)
        action_result_list.append(action_list_single_trail)

    # convert type of data
    ack_result_array = np.array(ack_result_list)
    tbs_result_array = np.array(tbs_result_list)
    sinr_feedback_result_array = np.array(sinr_feedback_result_list)
    mcs_result_array = np.array(mcs_result_list)
    sinr_result_array = np.array(sinr_result_list)
    action_result_array = np.array(action_result_list)

    file_path = "results/dqn_olla_LUT_sinr_quant_interval_" + str(feedback_interval) + \
        "_quant_bits_" + str(num_quant_bits) + "_num_trails_" + str(num_trails) + "_ack_weight_" + str(dqn_ack_weight) \
        + "_interference_result"".h5"
    olla_lut_file = h5py.File(file_path, "w")
    olla_lut_file.create_dataset(name="ACK", data=ack_result_array)
    olla_lut_file.create_dataset(name="TBS", data=tbs_result_array)
    olla_lut_file.create_dataset(name="SINR_feedback", data=sinr_feedback_result_array)
    olla_lut_file.create_dataset(name="MCS", data=mcs_result_array)
    olla_lut_file.create_dataset(name="SINR", data=sinr_result_array)
    olla_lut_file.create_dataset(name="num_trails", data=num_trails)
    olla_lut_file.create_dataset(name="num_time_slots", data=num_time_slots)
    olla_lut_file.create_dataset(name="feedback_interval", data=feedback_interval)
    olla_lut_file.create_dataset(name="action_index", data=action_result_list)
    olla_lut_file.close()


if __name__ == "__main__":
    dqn_olla_LUT_simulation_quant_sinr_feedback(num_trails=10, num_time_slots=500, feedback_interval=5, num_quant_bits=3, dqn_ack_weight=0)