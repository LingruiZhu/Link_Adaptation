import numpy as np
import matplotlib.pyplot as plt
import h5py


class ResultFileToPlot:
    def __init__(self, file_path:str, line_format:str, label:str) -> None:
        self.file_path = file_path
        self.line_format = line_format
        self.label = label


def read_result(file_path:str, num_time_slots_cut:int=None):
    result_file = h5py.File(file_path, "r")
    
    num_trails = int(result_file.get("num_trails")[()])
    num_time_slots = int(result_file.get("num_time_slots")[()])
    ack_array_raw = np.array(result_file.get("ACK"))
    cqi_array_raw = np.array(result_file.get("CQI"))
    tbs_array_raw = np.array(result_file.get("TBS"))
    mcs_array_raw = np.array(result_file.get("MCS"))
    sinr_array_raw = np.array(result_file.get("SINR"))
    action_array_raw = np.array(result_file.get("action_index"))
    sinr_offset_array_raw = np.array(result_file.get("sinr_offset"))
    sinr_feedback_array_raw = np.array(result_file.get("SINR_feedback"))

    ack_array = ack_array_raw[:, -num_time_slots_cut:]
    tbs_array = tbs_array_raw[:, -num_time_slots_cut:]
    mcs_array = mcs_array_raw[:, -num_time_slots_cut:]
    sinr_array = sinr_array_raw[:, -num_time_slots_cut:]
    action_array = action_array_raw[:, -num_time_slots_cut:]
    sinr_offset_array = sinr_offset_array_raw[:, -num_time_slots_cut:]
    sinr_feedback_array = sinr_feedback_array_raw[:, -num_time_slots_cut:]
    
    time_duration = 466.667e-6
    bandwidth = 2.16e6

    bler_avg = 1 - np.mean(ack_array, axis=0)
    tbs_avg = np.mean(tbs_array*ack_array, axis=0)
    data_rate_avg = tbs_avg / (time_duration*bandwidth)
    if result_file.get("CQI") is not None:
        cqi_array = cqi_array_raw[:, -num_time_slots_cut:]
        cqi_avg = np.mean(cqi_array, axis=0)
    else:
        cqi_avg = 0
    mcs_avg = np.mean(mcs_array, axis=0)
    sinr_avg = np.mean(sinr_array, axis=0)
    sinr_offset_avg = np.mean(sinr_offset_array, axis=0)
    
    if num_time_slots_cut != None:
        time_slot = np.arange(0, num_time_slots_cut)
    else:
        time_slot = np.arange(0, num_time_slots)
    return bler_avg, data_rate_avg, cqi_avg, mcs_avg, sinr_avg, sinr_offset_avg, time_slot


def plot_single_bler(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    bler_avg, _, _, _, _,_,time_slots = read_result(file_path, num_time_slots_cut)
    plt.plot(time_slots, bler_avg, line_format, label=label)


def plot_single_tputs(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    _, data_rate_avg, _, _, _, _, time_slots = read_result(file_path, num_time_slots_cut)
    plt.plot(time_slots, data_rate_avg, line_format, label=label)


def plot_single_cqi(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    _, _, cqi_avg, _, _, _, time_slots = read_result(file_path, num_time_slots_cut)
    plt.plot(time_slots, cqi_avg, line_format, label=label)


def plot_single_mcs(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    _, _, _, mcs_avg, _, _, time_slots  = read_result(file_path, num_time_slots_cut)
    plt.plot(time_slots, mcs_avg, line_format, label=label)


def plot_single_estimated_snr(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    _, _, _, _, sinr_avg, _, time_slots = read_result(file_path, num_time_slots_cut)
    plt.plot(time_slots, sinr_avg, line_format, label=label)
    

def plot_singel_sinr_offset(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    _, _, _, _, _, sinr_offset_avg, time_slots = read_result(file_path, num_time_slots_cut)
    plt.plot(time_slots, sinr_offset_avg, line_format, label=label)


def plot_single_tputs_cdf(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    _, data_rate_avg, _, _, _, _, time_slots = read_result(file_path, num_time_slots_cut)
    counts, bins_count = np.histogram(data_rate_avg, bins=100)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, line_format, label=label)


def plot_single_bler_cdf(file_path:str, line_format:str, label:str, num_time_slots_cut:int=None):
    bler_avg, _, _, _, _, _, time_slots = read_result(file_path, num_time_slots_cut)
    counts, bins_count = np.histogram(bler_avg, bins=100)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, line_format, label=label)



def plot_tputs(result_list:list, num_time_slots_cut:int=None):
    plt.figure()
    for result in result_list:
        plot_single_tputs(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("Time")
    plt.ylabel("Throughputs")
    plt.legend()
    plt.grid()


def plot_bler(result_list:list, num_time_slots_cut:int=None):
    plt.figure()
    for result in result_list:
        plot_single_bler(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("Time")
    plt.ylabel("BLER")
    plt.legend()
    plt.grid()


def plot_CQI():
    # olla results
    olla_result_file = "simulation_LUT_result/olla_LUT_result.h5"
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


def plot_sinr(result_list:list, num_time_slots_cut:int=None):
    plt.figure()
    for result in result_list:
        plot_single_estimated_snr(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("Time")
    plt.ylabel("SINR")
    plt.legend()
    plt.grid()


def plot_tputs_cdf(result_list:list, num_time_slots_cut:int=None):
    plt.figure()
    for result in result_list:
        plot_single_tputs_cdf(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("tputs")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()


def plot_bler_cdf(result_list:list, num_time_slots_cut:int=None):
    plt.figure()
    for result in result_list:
        plot_single_bler_cdf(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("BLER")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()


def plot_sinr_offset(result_list:list, num_time_slots_cut:int=None):
    plt.figure()
    for result in result_list:
        plot_singel_sinr_offset(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("Time")
    plt.ylabel("SINR offset")
    plt.legend()
    plt.grid()


    
if __name__ == "__main__":
    olla_interval_1 = ResultFileToPlot(file_path="simulation_LUT_result/olla_LUT_sinr_interval_1_interference_result.h5",\
                                       line_format="b-x", label="OLLA interval 1")
    olla_interval_5 = ResultFileToPlot(file_path="simulation_LUT_result/olla_LUT_sinr_interval_5_interference_result.h5",\
                                       line_format="g-o", label="OLLA interval 5")
    olla_interval_5_lstm = ResultFileToPlot(file_path="simulation_LUT_result/olla_LUT_sinr_interval_5_lstm_interference_result.h5", \
                                        line_format="r-d", label="olla lstm interval 5")
    olla_interval_5_quant_sinr_4_bits = ResultFileToPlot(file_path="experiment_simulation_quantized/results/olla_LUT_sinr_interval_5_quant_bits_4_interference_result.h5",\
                                                line_format="c-s", label="No prediction - 4 bits")
    olla_interval_5_quant_lstm_sinr_4_bits = ResultFileToPlot(file_path="experiment_simulation_quantized/results/olla_LUT_sinr_interval_5_lstm_quant_bits_4_interference_result.h5",
                                        line_format="g-s", label="LSTM prediction - 4 bits")
    illa_interval_5_quant_3_bits = ResultFileToPlot(file_path="experiment_simulation_quantized/results/olla_LUT_sinr_interval_5_quant_bits_3_interference_result.h5",\
                                        line_format="b-x", label="ILLA - No prediction - 3 bits")
    illa_interval_5_quant_lstm_3_bits = ResultFileToPlot(file_path="experiment_simulation_quantized/results/olla_LUT_sinr_interval_5_lstm_quant_bits_3_interference_result.h5", \
                                        line_format="r-x", label="ILLA - LSTM prediction - 3 bits")
    olla_interval_5_quant_3_bits = ResultFileToPlot(file_path="experiment_simulation_quantized/results/new_olla_LUT_sinr_interval_5_quant_bits_3_interference_result.h5", \
                                        line_format="m-s", label="OLLA - No prediction - 3 bits")
    olla_interval_5_quant_lstm_3_bits = ResultFileToPlot(file_path="experiment_simulation_quantized/results/new_olla_LUT_sinr_interval_5_lstm_quant_bits3_interference_result.h5", \
                                        line_format="c-s", label="OLLA - LSTM prediction - 3 bits")
    dqn_olla_interval_5_quant_3_bits = ResultFileToPlot(file_path="experiment_simulation_quantized/results/dqn_olla_LUT_sinr_quant_interval_5_quant_bits_3_num_trails_10_interference_result.h5", \
                                        line_format="g-d", label="DQN - No prediction - 3 bits")
    
    
    dqn_olla_ack_weights_0 = ResultFileToPlot(file_path="experiment_simulation_quantized/results/dqn_olla_LUT_sinr_quant_interval_5_quant_bits_3_num_trails_10_ack_weight_0_interference_result.h5", \
                                        line_format="g-x", label="ack_weights = 0")
    dqn_olla_ack_weights_1 = ResultFileToPlot(file_path="experiment_simulation_quantized/results/dqn_olla_LUT_sinr_quant_interval_5_quant_bits_3_num_trails_10_ack_weight_1_interference_result.h5", \
                                        line_format="c-x", label="ack_weights = 1")
    dqn_olla_ack_weights_3 = ResultFileToPlot(file_path="experiment_simulation_quantized/results/dqn_olla_LUT_sinr_quant_interval_5_quant_bits_3_num_trails_10_ack_weight_3_interference_result.h5", \
                                        line_format="b-x", label="ack_weights = 3")
    dqn_olla_ack_weights_4 = ResultFileToPlot(file_path="experiment_simulation_quantized/results/dqn_olla_LUT_sinr_quant_interval_5_quant_bits_3_num_trails_10_ack_weight_4_interference_result.h5", \
                                        line_format="r-x", label="ack_weights = 4")
    
    
    olla_benchmark = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_ack_weight_0_num_trails_20_cqi_interval_5_quant_bits_4_delay_0.h5", \
                                        line_format="r-x", label="OLLA")
    DQN_v2 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_DQN_v2_ack_weight_0_num_trails_20_cqi_interval_5_quant_bits_4_delay_0.h5", \
                                        line_format="g-x", label="OLLA DQN")
    
    olla_benchmark_delay_2 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_ack_weight_0_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_True_correct_sinr_reward_training_long.h5", \
                                        line_format="r-s", label="OLLA delay 2")
    DQN_v2_delay_2 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_DQN_v2_ack_weight_0_num_time_slots_6_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_True_reward_and_more_actions.h5", \
                                        line_format="g-s", label="OLLA DQN delay 2")
    
    
    result_list = []
    result_list.append(olla_benchmark_delay_2)
    result_list.append(DQN_v2_delay_2)
     
    num_time_slots_plot = 400
    
    plot_bler(result_list, num_time_slots_plot)
    plot_tputs(result_list, num_time_slots_plot)
    plot_sinr(result_list, num_time_slots_plot)
    plot_tputs_cdf(result_list, num_time_slots_plot)
    plot_bler_cdf(result_list, num_time_slots_plot)
    plot_sinr_offset(result_list, num_time_slots_plot)
    plt.show()