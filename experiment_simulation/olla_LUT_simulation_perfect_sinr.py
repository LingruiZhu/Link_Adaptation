import numpy as np
import h5py

import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

from la_agent.olla_agent import OuterLoopLinkAdaptation
from link_envinorment import LutEnvironment
from InterferenceToy import InterferenceToy



def olla_lut_simulation_perfect_single_trail(num_time_slots:int, cqi_intervals:int):
    # TODO: insert CQI intervals to simulation
    # initialize objects
    olla_agent = OuterLoopLinkAdaptation(bler_target=0.1, 
                                         data_file="BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                         olla_step_size=0.05)
    lut_link_env = LutEnvironment(ebno_db=10.5)
    interference1 = InterferenceToy(tx_power=0.1, h_factor=0.3, interval=10, period=2)
    interference2 = InterferenceToy(tx_power=0.1, h_factor=0.2, interval=6, period=3)
    lut_link_env.add_interference_noise(interference1)
    lut_link_env.add_interference_noise(interference2)
    lut_link_env.set_interference_mode(True)

    modu_order = 2
    code_rate = 602/1024

    # initialize result list
    ack_list = list()
    tbs_list = list()
    sinr_list = list()
    mcs_list = list()
    sinr_eff_list = list()

    for time_index in range(num_time_slots):
        ack, tbs_size, _, eff_sinr = lut_link_env.step(modu_order, code_rate)
        # update cqi every num_time slots
        if time_index % cqi_intervals == 0:
            sinr_feedback = eff_sinr
        mcs_index, code_rate, modu_order, sinr_eff = olla_agent.determine_mcs_action_from_sinr(sinr_feedback, ack)


        # save result
        ack_list.append(ack)
        tbs_list.append(tbs_size)
        sinr_list.append(sinr_feedback)
        mcs_list.append(mcs_index)
        sinr_eff_list.append(sinr_eff)
    
    return ack_list, tbs_list, sinr_list, mcs_list, sinr_eff_list


def olla_LUT_simulation_sinr_feedback(num_trails, num_time_slots, feedback_interval):
    ack_result_list = list()
    tbs_result_list = list()
    sinr_feedback_result_list = list()
    mcs_result_list = list()
    sinr_result_list = list()

    for idx_train in range(num_trails):
        ack_list_single_trail, tbs_list_single_trail, sinr_feedback_list, mcs_list_single_trail, sinr_list_single_trail = \
            olla_lut_simulation_perfect_single_trail(num_time_slots, feedback_interval)
        ack_result_list.append(ack_list_single_trail)
        tbs_result_list.append(tbs_list_single_trail)
        sinr_feedback_result_list.append(sinr_feedback_list)
        mcs_result_list.append(mcs_list_single_trail)
        sinr_result_list.append(sinr_list_single_trail)

    # convert type of data
    ack_result_array = np.array(ack_result_list)
    tbs_result_array = np.array(tbs_result_list)
    sinr_feedback_result_array = np.array(sinr_feedback_result_list)
    mcs_result_array = np.array(mcs_result_list)
    sinr_result_array = np.array(sinr_result_list)

    file_path = "simulation_LUT_result/olla_LUT_sinr_interval_" + str(feedback_interval) +"_interference_result.h5"
    olla_lut_file = h5py.File(file_path, "w")
    olla_lut_file.create_dataset(name="ACK", data=ack_result_array)
    olla_lut_file.create_dataset(name="TBS", data=tbs_result_array)
    olla_lut_file.create_dataset(name="SINR_feedback", data=sinr_feedback_result_array)
    olla_lut_file.create_dataset(name="MCS", data=mcs_result_array)
    olla_lut_file.create_dataset(name="SINR", data=sinr_result_array)
    olla_lut_file.create_dataset(name="num_trails", data=num_trails)
    olla_lut_file.create_dataset(name="num_time_slots", data=num_time_slots)
    olla_lut_file.create_dataset(name="feedback_interval", data=feedback_interval)
    olla_lut_file.close()


if __name__ == "__main__":
    olla_LUT_simulation_sinr_feedback(num_trails=100, num_time_slots=500, feedback_interval=1)