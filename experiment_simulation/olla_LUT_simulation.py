import numpy as np
import h5py


import sys
sys.path.append("/home/zhu/Codes/link_adaptation")


from la_agent.olla_agent import OuterLoopLinkAdaptation
from link_envinorment import LutEnvironment


def olla_lut_simulation_single_trail(num_time_slots:int, cqi_intervals:int):
    # TODO: insert CQI intervals to simulation
    # initialize objects
    olla_agent = OuterLoopLinkAdaptation(bler_target=0.1, 
                                         data_file="BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                         olla_step_size=0.05)
    lut_link_env = LutEnvironment(ebno_db=10)
    modu_order = 2
    code_rate = 602/1024

    # initialize result list
    ack_list = list()
    tbs_list = list()
    cqi_list = list()
    mcs_list = list()
    sinr_eff_list = list()

    for time_index in range(num_time_slots):
        ack, tbs_size, cqi_rt, _ = lut_link_env.step(modu_order, code_rate)
        # update cqi every num_time slots
        if time_index % cqi_intervals == 0:
            cqi = cqi_rt
        mcs_index, code_rate, modu_order, sinr_eff = olla_agent.determine_mcs_action(cqi, ack)
        # save result
        ack_list.append(ack)
        tbs_list.append(tbs_size)
        cqi_list.append(cqi)
        mcs_list.append(mcs_index)
        sinr_eff_list.append(sinr_eff)
    
    return ack_list, tbs_list, cqi_list, mcs_list, sinr_eff_list


def olla_LUT_simulation(num_trails, num_time_slots, cqi_interval):
    ack_result_list = list()
    tbs_result_list = list()
    cqi_result_list = list()
    mcs_result_list = list()
    sinr_result_list = list()

    for idx_train in range(num_trails):
        ack_list_single_trail, tbs_list_single_trail, cqi_list_single_trail, mcs_list_single_trail, sinr_list_single_trail = \
            olla_lut_simulation_single_trail(num_time_slots, cqi_interval)
        ack_result_list.append(ack_list_single_trail)
        tbs_result_list.append(tbs_list_single_trail)
        cqi_result_list.append(cqi_list_single_trail)
        mcs_result_list.append(mcs_list_single_trail)
        sinr_result_list.append(sinr_list_single_trail)

    # convert type of data
    ack_result_array = np.array(ack_result_list)
    tbs_result_array = np.array(tbs_result_list)
    cqi_result_array = np.array(cqi_result_list)
    mcs_result_array = np.array(mcs_result_list)
    sinr_result_array = np.array(sinr_result_list)

    file_path = "simulation_LUT_result/olla_LUT_cqi_interval_" + str(cqi_interval) +"_result.h5"
    olla_lut_file = h5py.File(file_path, "w")
    olla_lut_file.create_dataset(name="ACK", data=ack_result_array)
    olla_lut_file.create_dataset(name="TBS", data=tbs_result_array)
    olla_lut_file.create_dataset(name="CQI", data=cqi_result_array)
    olla_lut_file.create_dataset(name="MCS", data=mcs_result_array)
    olla_lut_file.create_dataset(name="SINR", data=sinr_result_array)
    olla_lut_file.create_dataset(name="num_trails", data=num_trails)
    olla_lut_file.create_dataset(name="num_time_slots", data=num_time_slots)
    olla_lut_file.create_dataset(name="CQI_interval", data=cqi_interval)
    olla_lut_file.close()


if __name__ == "__main__":
    olla_LUT_simulation(num_trails=100, num_time_slots=500, cqi_interval=1)
