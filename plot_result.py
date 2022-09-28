import numpy as np
import matplotlib.pyplot as plt
import h5py


def read_result(file_path:str):
    result_file = h5py.File(file_path, "r")
    
    num_trails = int(result_file.get("num_trails")[()])
    num_time_slots = int(result_file.get("num_time_slots")[()])
    ack_array = np.array(result_file.get("ACK"))
    cqi_array = np.array(result_file.get("CQI"))
    tbs_array = np.array(result_file.get("TBS"))
    mcs_array = np.array(result_file.get("MCS"))
    sinr_array = np.array(result_file.get("SINR"))
    
    time_duration = 466.667e-6
    bandwidth = 2.16e6

    bler_avg = 1 - np.mean(ack_array, axis=0)
    tbs_avg = np.mean(tbs_array*ack_array, axis=0)
    data_rate_avg = tbs_avg / (time_duration*bandwidth)
    cqi_avg = np.mean(cqi_array, axis=0)
    mcs_avg = np.mean(mcs_array, axis=0)
    sinr_avg = np.mean(sinr_array, axis=0)

    time_slot = np.arange(0, num_time_slots)
    return bler_avg, data_rate_avg, cqi_avg, mcs_avg, sinr_avg, time_slot


def plot_single_bler(file_path:str, line_format:str, label:str):
    bler_avg, _, _, _, _,time_slots = read_result(file_path)
    plt.plot(time_slots, bler_avg, line_format, label=label)


def plot_single_tputs(file_path:str, line_format:str, label:str):
    _, data_rate_avg, _, _, _, time_slots = read_result(file_path)
    plt.plot(time_slots, data_rate_avg, line_format, label=label)


def plot_single_cqi(file_path:str, line_format:str, label:str):
    _, _, cqi_avg, _, _,time_slots = read_result(file_path)
    plt.plot(time_slots, cqi_avg, line_format, label=label)


def plot_single_mcs(file_path:str, line_format:str, label:str):
    _, _, _, mcs_avg, _, time_slots  = read_result(file_path)
    plt.plot(time_slots, mcs_avg, line_format, label=label)


def plot_single_estimated_snr(file_path:str, line_format:str, label:str):
    _, _, _, _, sinr_avg, time_slots = read_result(file_path)
    plt.plot(time_slots, sinr_avg, line_format, label=label)


def plot_tputs():
    # olla results
    # olla_result_file = "simulation_LUT_result/olla_LUT_result.h5"
    olla_result_file = "simulation_LUT_result/olla_LUT_cqi_interval_1_result.h5"
    plt.figure()
    plot_single_tputs(olla_result_file, line_format="b-x", label="olla")
    plt.xlabel("Time")
    plt.ylabel("Throughputs")
    plt.grid()


def plot_bler():
    # olla results
    # olla_result_file_interval_5 = "simulation_LUT_result/olla_LUT_cqi_interval_5_result.h5"
    olla_result_file = "simulation_LUT_result/olla_LUT_cqi_interval_1_result.h5"
    plt.figure()
    plot_single_bler(olla_result_file, line_format="b-x", label="olla")
    plt.xlabel("Time")
    plt.ylabel("BLER")
    plt.grid()


def plot_CQI():
    # olla results
    # olla_result_file = "simulation_LUT_result/olla_LUT_result.h5"
    olla_result_file = "simulation_LUT_result/olla_LUT_cqi_interval_1_result.h5"
    plt.figure()
    plot_single_cqi(olla_result_file, line_format="b-x", label="olla")
    plt.xlabel("Time")
    plt.ylabel("CQI")
    plt.grid()


def plot_MCS():
    olla_result_file = "simulation_LUT_result/olla_LUT_cqi_interval_1_result.h5"
    plt.figure()
    plot_single_mcs(olla_result_file, line_format="b-x", label="olla")
    plt.xlabel("Time")
    plt.ylabel("MCS")
    plt.grid()


def plot_sinr():
    olla_result_file = "simulation_LUT_result/olla_LUT_cqi_interval_1_result.h5"
    plt.figure()
    plot_single_estimated_snr(olla_result_file, line_format="b-x", label="olla")
    plt.xlabel("Time")
    plt.ylabel("Estimated sinr")
    plt.grid()

    
    

if __name__ == "__main__":
    plot_bler()
    plot_tputs()
    plot_CQI()
    plot_MCS()
    plot_sinr()
    plt.show()