import numpy as np
import h5py

from olla_agent import OuterLoopLinkAdaptation
from link_envinorment import LinkEnvironment


def olla_simulation(num_trails:int, num_time_slots):
    ack_result_list = list()
    tbs_result_list = list()
    cqi_result_list = list()
    mcs_result_list = list()

    for i in range(num_trails):
        ack_list_single_trail, tbs_list_single_trail, cqi_list_single_trail, mcs_list_single_trail = olla_simulation_single_trail(num_time_slots=num_time_slots)
        ack_result_list.append(ack_list_single_trail)
        tbs_result_list.append(tbs_list_single_trail)
        cqi_result_list.append(cqi_list_single_trail)
        mcs_result_list.append(mcs_list_single_trail)
    
    ack_result_array = np.array(ack_result_list)
    tbs_result_array = np.array(tbs_result_list)
    cqi_result_array = np.array(cqi_result_list)
    mcs_result_array = np.array(mcs_result_list)

    # save the result to .h5 file
    result_file = h5py.File("simulation_results/olla_result.h5", "w")
    result_file.create_dataset(name="ACK", data=ack_result_array)
    result_file.create_dataset(name="TBS", data=tbs_result_array)
    result_file.create_dataset(name="CQI", data=cqi_result_array)
    result_file.create_dataset(name="MCS", data=mcs_result_array)
    result_file.create_dataset(name="num_trails", data=num_trails)
    result_file.create_dataset(name="num_time_slots", data=num_time_slots)
    result_file.close()


def olla_simulation_single_trail(num_time_slots:int):
    # initialize objects
    olla_agent = OuterLoopLinkAdaptation(bler_target=0.1, 
                                         data_file="BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                         olla_step_size=0.1)
    link_env = LinkEnvironment()
    modu_order = 2
    code_rate = 602/1024

    # initialize result list
    ack_list = list()
    tbs_list = list()
    cqi_list = list()
    mcs_list = list()

    for time_index in range(num_time_slots):
        ack, tbs_size, cqi = link_env.step(modu_order, code_rate)
        mcs_index, code_rate, modu_order = olla_agent.determine_mcs_action(cqi, ack)

        # save result
        ack_list.append(ack)
        tbs_list.append(tbs_size)
        cqi_list.append(cqi)
        mcs_list.append(mcs_index)
    
    return ack_list, tbs_list, cqi_list, mcs_list


if __name__ == "__main__":
    olla_simulation(num_trails=100, num_time_slots=500)