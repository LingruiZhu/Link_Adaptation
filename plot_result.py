import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib                                                   


class ResultFileToPlot:
    def __init__(self, file_path:str or list, line_format:str, label:str) -> None:
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
    
    bler_avg_per_ue = 1 - np.mean(ack_array, axis=1)
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
    return bler_avg, data_rate_avg, cqi_avg, mcs_avg, sinr_avg, sinr_offset_avg, time_slot, bler_avg_per_ue


def read_result_multiple_file(file_path_list:list, num_time_slots_cut:int=None, list_length:int=20):
    bler_avg_list:list = list()
    data_rate_avg_list:list = list()
    cqi_avg_list:list = list()
    msc_avg_list:list = list()
    sinr_avg_list:list = list()
    sinr_offset_avg_list:list = list()
    bler_avg_per_ue_list = list()
    if num_time_slots_cut != None:
        time_slot = np.arange(0, num_time_slots_cut)
    else:
        time_slot = np.arange(0, num_time_slots_cut)
    
    for file_path in file_path_list:
        bler_avg, data_rate_avg, cqi_avg, mcs_avg, sinr_avg, sinr_offset_avg, _ , bler_avg_per_ue  = read_result(file_path, num_time_slots_cut)
        bler_avg_list.append(bler_avg)
        data_rate_avg_list.append(data_rate_avg)
        cqi_avg_list.append(cqi_avg)
        msc_avg_list.append(mcs_avg)
        sinr_avg_list.append(sinr_avg)
        sinr_offset_avg_list.append(sinr_offset_avg)
        bler_avg_per_ue_list.append(bler_avg_per_ue)
    
    if list_length == None:
        list_length = len(bler_avg_avg)
        
    bler_avg_per_UE = np.mean(np.array(bler_avg_list[:list_length]), axis=1)
    bler_avg_avg = np.mean(np.array(bler_avg_list[:list_length]), axis=0)
    data_rate_avg_avg = np.mean(np.array(data_rate_avg_list[:list_length]), axis=0)
    cqi_avg_avg = np.mean(np.array(cqi_avg_list[:list_length]), axis=0)
    mcs_avg_avg = np.mean(np.array(msc_avg_list[:list_length]), axis=0)
    sinr_avg_avg = np.mean(np.array(sinr_avg_list[:list_length]), axis=0)
    sinr_offset_avg_avg = np.mean(np.array(sinr_offset_avg_list[:list_length]), axis=0)
    
    return bler_avg_avg, data_rate_avg_avg, cqi_avg_avg, mcs_avg_avg, sinr_avg_avg, sinr_offset_avg_avg, time_slot, bler_avg_per_UE
    

def plot_single_bler(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None, plot_axes:matplotlib.axes=None):
    if isinstance(file_path, str):
        bler_avg, _, _, _, _,_,time_slots, _ = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        bler_avg, _, _, _, _,_,time_slots, = read_result_multiple_file(file_path, num_time_slots_cut)
    if plot_axes == None:
        plt.plot(time_slots, bler_avg, line_format, label=label)
    else:
        plot_axes.plot(time_slots, bler_avg, line_format, label=label)


def plot_single_tputs(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None, plot_axes:matplotlib.axes=None):
    if isinstance(file_path, str):
        _, data_rate_avg, _, _, _, _, time_slots, _ = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        _, data_rate_avg, _, _, _, _, time_slots = read_result_multiple_file(file_path, num_time_slots_cut)
    if plot_axes == None:
        plt.plot(time_slots, data_rate_avg, line_format, label=label)
    else:
        plot_axes.plot(time_slots, data_rate_avg, line_format, label=label)

def plot_single_cqi(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None):
    if isinstance(file_path, str):
        _, _, cqi_avg, _, _, _, time_slots, _ = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        _, _, cqi_avg, _, _, _, time_slots = read_result_multiple_file(file_path, num_time_slots_cut)
    plt.plot(time_slots, cqi_avg, line_format, label=label)


def plot_single_mcs(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None):
    if isinstance(file_path, str):
        _, _, _, mcs_avg, _, _, time_slots, _  = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        _, _, _, mcs_avg, _, _, time_slots  = read_result_multiple_file(file_path, num_time_slots_cut)
    plt.plot(time_slots, mcs_avg, line_format, label=label)


def plot_single_estimated_snr(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None):
    if isinstance(file_path, str):
        _, _, _, _, sinr_avg, _, time_slots, _ = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        _, _, _, _, sinr_avg, _, time_slots = read_result_multiple_file(file_path, num_time_slots_cut)
    plt.plot(time_slots, sinr_avg, line_format, label=label)
    

def plot_singel_sinr_offset(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None):
    if isinstance(file_path, str):
        _, _, _, _, _, sinr_offset_avg, time_slots, _ = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        _, _, _, _, _, sinr_offset_avg, time_slots = read_result_multiple_file(file_path, num_time_slots_cut)
    plt.plot(time_slots, sinr_offset_avg, line_format, label=label)


def plot_single_tputs_cdf(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None, plot_axes:matplotlib.axes=None):
    if isinstance(file_path, str):
        _, data_rate_avg, _, _, _, _, time_slots, _ = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        _, data_rate_avg, _, _, _, _, time_slots = read_result_multiple_file(file_path, num_time_slots_cut)
    counts, bins_count = np.histogram(data_rate_avg, bins=100)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    color = line_format[0]
    if plot_axes == None:
        plt.plot(bins_count[1:], cdf, color, label=label)
    else:
        plot_axes.plot(bins_count[1:], cdf, color, label=label)
        

def plot_single_bler_cdf(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None, plot_axes:matplotlib.axes=None):
    if isinstance(file_path, str):
        bler_avg, _, _, _, _, _, time_slots, _ = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        bler_avg, _, _, _, _, _, time_slots = read_result_multiple_file(file_path, num_time_slots_cut)
    counts, bins_count = np.histogram(bler_avg, bins=100)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    color = line_format[0]
    if plot_axes == None:
        plt.plot(bins_count[1:], cdf, color, label=label)
    else:
        plot_axes.plot(bins_count[1:], cdf, color, label=label)


def plot_single_bler_UE_cdf(file_path:str or list, line_format:str, label:str, num_time_slots_cut:int=None, plot_axes:matplotlib.axes=None):
    if isinstance(file_path, str):
        _ , _, _, _, _, _, time_slots, bler_avg_per_ue = read_result(file_path, num_time_slots_cut)
    elif isinstance(file_path, list):
        bler_avg, _, _, _, _, _, time_slots = read_result_multiple_file(file_path, num_time_slots_cut)
    counts, bins_count = np.histogram(bler_avg_per_ue, bins=100)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    color = line_format[0]
    if plot_axes == None:
        plt.plot(bins_count[1:], cdf, color, label=label)
    else:
        plot_axes.plot(bins_count[1:], cdf, color, label=label)
        

def plot_tputs(result_list:list, num_time_slots_cut:int=None, save_figure:bool=False, save_folder:str=None, plot_axes:matplotlib.axes=None):
    plt.figure()
    for result in result_list:
        plot_single_tputs(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut, plot_axes=plot_axes)
    plt.xlabel("Time")
    plt.ylabel("Throughputs")
    plt.legend()
    plt.grid()
    if save_figure:
        fig_path = save_folder + "Tputs.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    

def plot_tputs_subplots(result_list:list, num_time_slots_cut:int=None, save_figure:bool=False, save_folder:str=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    for result in result_list:
        plot_single_tputs(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut, plot_axes=ax1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Spectrual Efficiency (bps/Hz)")
    ax1.legend()
    ax1.grid()
    
    for result in result_list:
        plot_single_tputs_cdf(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut, plot_axes=ax2)
    ax2.set_xlabel("Spectrual Efficiency (bps/Hz)")
    ax2.set_ylabel("CDF")
    ax2.legend()
    ax2.grid()

    if save_figure:
        fig_path = save_folder + "SE_both.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    



def plot_bler(result_list:list, num_time_slots_cut:int=None, save_figure:bool=False, save_folder:str=None):
    plt.figure()
    for result in result_list:
        plot_single_bler(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("Time")
    plt.ylabel("BLER")
    plt.legend()
    plt.grid()
    if save_figure:
        fig_path = save_folder + "BLER.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")


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


def plot_sinr(result_list:list, num_time_slots_cut:int=None, save_figure:bool=False, save_folder:str=None):
    plt.figure()
    for result in result_list:
        plot_single_estimated_snr(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("Time")
    plt.ylabel("SINR")
    plt.legend()
    plt.grid()
    if save_figure:
        fig_path = save_folder + "SINR.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")


def plot_tputs_cdf(result_list:list, num_time_slots_cut:int=None, save_figure:bool=False, save_folder:str=None):
    plt.figure()
    for result in result_list:
        plot_single_tputs_cdf(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.axhline(y=0.1, c="red", linestyle="--")
    plt.xlabel("Throughput")
    plt.ylabel("CDF")
    plt.text(2.4, 0.12, '10th percentile', fontsize=12, color = 'r')
    plt.legend()
    plt.grid()
    if save_figure:
        fig_path = save_folder + "tputs_cdf.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")


def plot_bler_cdf(result_list:list, num_time_slots_cut:int=None, save_figure:bool=False, save_folder:str=None):
    plt.figure()
    for result in result_list:
        plot_single_bler_cdf(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("BLER")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.axhline(y=0.9, c="red", linestyle="--")
    plt.text(0.25, 0.88, '90th percentile', fontsize=12, color = 'r')
    if save_figure:
        fig_path = save_folder + "bler_cdf.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")
        

def plot_bler_per_ue_cdf(result_list:list, num_time_slots_cut:int=None, save_figure:bool=False, save_folder:str=None):
    plt.figure()
    for result in result_list:
        plot_single_bler_UE_cdf(result.file_path, line_format=result.line_format, label=result.label, num_time_slots_cut=num_time_slots_cut)
    plt.xlabel("BLER (per UE)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.axhline(y=0.9, c="red", linestyle="--")
    plt.text(0.25, 0.88, '90th percentile', fontsize=12, color = 'r')
    if save_figure:
        fig_path = save_folder + "bler_cdf.pdf"
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")


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
    dqn_olla_ack_weights_4 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_gamma_0.75_logistic_link_abstractor.h5", \
                                        line_format="r-x", label="ack_weights = 4")
    
    
    olla_benchmark = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_bench_mark.h5", \
                                        line_format="r-x", label="OLLA")
    DQN_v2 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_DQN_v2_ack_weight_0_num_trails_20_cqi_interval_5_quant_bits_4_delay_0.h5", \
                                        line_format="g-x", label="OLLA DQN")
    
    DQN_result_path_list = list()
    DQN_result_path_list.append("experiment_simulation_quantized/results_v1/gamma_0/OLLA_DQN_v1_ack_weight_0_num_time_slots_4_num_trails_3_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_reward_normalized_instant_complete_01.h5")
    DQN_result_path_list.append("experiment_simulation_quantized/results_v1/gamma_0/OLLA_DQN_v1_ack_weight_0_num_time_slots_4_num_trails_3_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_reward_normalized_instant_complete_02.h5")
    DQN_result_path_list.append("experiment_simulation_quantized/results_v1/gamma_0/OLLA_DQN_v1_ack_weight_0_num_time_slots_4_num_trails_3_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_reward_normalized_instant_complete_03.h5")
    DQN_result_path_list.append("experiment_simulation_quantized/results_v1/gamma_0/OLLA_DQN_v1_ack_weight_0_num_time_slots_4_num_trails_3_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_reward_normalized_instant_complete_04.h5")
    DQN_result_path_list.append("experiment_simulation_quantized/results_v1/gamma_0/OLLA_DQN_v1_ack_weight_0_num_time_slots_4_num_trails_3_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_reward_normalized_instant_complete_05.h5")
    DQN_result_path_list.append("experiment_simulation_quantized/results_v1/gamma_0/OLLA_DQN_v1_ack_weight_0_num_time_slots_4_num_trails_3_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_reward_normalized_instant_complete_01.h5")
    DQN_result_path_list.append("experiment_simulation_quantized/results_v1/gamma_0/OLLA_DQN_v1_ack_weight_0_num_time_slots_4_num_trails_3_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_reward_normalized_instant_complete_02.h5")

     
    
    Result_A = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_simulation_1220.h5", \
                                        line_format="r-x", label="OLLA")
    Result_B = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/NOLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_gamma_0.75_matched_parameter.h5",
                                        line_format="b-o", label="NOLLA")
    Result_C = ResultFileToPlot(file_path=DQN_result_path_list, \
                                line_format="y-s", label="DQN")
    Result_D = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/NOLLA_bp_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_gamma_0.75_LA+LRLA.h5", \
                                        line_format="b-x", label="NOLLA_bp")
    
    
    
    Nolla_result_1 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.6.h5",
                                      line_format="g-d", label="alpha_up=0.6")
    Nolla_result_2 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.65.h5",
                                      line_format="r-x", label="alpha_up=0.65")
    Nolla_result_3 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.70.h5",
                                      line_format="b-s", label="alpha_up=0.70")
    Nolla_result_4 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.75.h5",
                                      line_format="y-+", label="alpha_up=0.75")
    Nolla_result_5 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.80.h5",
                                      line_format="m-h", label="Nolla alpha_up=0.80")
    Nolla_result_6 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.85.h5",
                                      line_format="c-p", label="alpha_up=0.85")
    Nolla_result_7 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.90.h5",
                                      line_format="k-1", label="alpha_up=0.90")
    Nolla_result_8 = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.95.h5",
                                      line_format="g--d", label="alpha_up=0.95")
    Olla_result = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/OLLA_bench_mark.h5",
                                   line_format="r--x", label="OLLA")
    
    # static SINR
    OLLA_static_sinr = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/OLLA_bench_mark.h5",
                                   line_format="r--x", label="OLLA")
    NOLLA_static_sinr_matched = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.80.h5",
                                      line_format="b--o", label="NOLLA")
    NOLLA_static_sinr_unmatched = ResultFileToPlot(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.95.h5",
                                      line_format="g--d", label="NOLLA-unmatched")
    
    
    # With Interference
    OLLA_with_interference = ResultFileToPlot(file_path="experiment_simulation_quantized/results_v2/OLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_True_variant_sinr_False_gamma_0.75_0302.h5",
                                   line_format="r--x", label="OLLA")
    NOLLA_with_interference = ResultFileToPlot(file_path="/home/zhu/Codes/link_adaptation/experiment_simulation_quantized/results_v2/NOLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_True_variant_sinr_False_gamma_0.75_0302.h5",
                                   line_format="b--o", label="NOLLA")
    
    # Variant_SINR
    OLLA_variant_sinr = ResultFileToPlot(file_path="experiment_simulation_quantized/results_for_paper/variant_SINR/OLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_True_gamma_0.75_0228.h5",
                                   line_format="r--x", label="OLLA")
    NOLLA_variant_sinr = ResultFileToPlot(file_path="experiment_simulation_quantized/results_for_paper/variant_SINR/NOLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_True_gamma_0.75_0228.h5",
                                   line_format="b--o", label="NOLLA")
    
    # Low BLER results
    OLLA_low_bler = ResultFileToPlot(file_path="experiment_simulation_quantized/low_bler_results/OLLA_ack_weight_0_num_time_slots_4_num_trails_100_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_gamma_0.75_low_bler_003.h5",
                                     line_format="r--x", label="OLLA")
    NOLLA_low_bler = ResultFileToPlot(file_path="experiment_simulation_quantized/low_bler_results/NOLLA_ack_weight_0_num_time_slots_4_num_trails_100_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_gamma_0.75_low_bler_003.h5",
                                      line_format="b--o", label="NOLLA")
    
    
    NOLLA_per_UE = ResultFileToPlot(file_path="experiment_simulation_quantized/results_per_ue_bler/NOLLA_ack_weight_0_num_time_slots_4_num_trails_1000_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_gamma_0.75_per_UE_june_6666.h5",
                                   line_format="b--o", label="NOLLA")
    OLLA_per_UE = ResultFileToPlot(file_path="experiment_simulation_quantized/results_per_ue_bler/OLLA_ack_weight_0_num_time_slots_4_num_trails_1000_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_variant_sinr_False_gamma_0.75_per_UE_june_6666.h5",
                                   line_format="r--x", label="OLLA")

    result_list = []
    
    # result_list.append(olla_benchmark)
    # result_list.append(NOLLA_static_sinr_matched)
    # result_list.append(Result_C)
        
    # result_list.append(Nolla_result_1)
    # result_list.append(Nolla_result_2)
    # result_list.append(Nolla_result_3)
    # result_list.append(Nolla_result_4)
    # result_list.append(Nolla_result_5)
    # result_list.append(Nolla_result_6)
    # result_list.append(Nolla_result_7)
    # result_list.append(Nolla_result_8)
    # result_list.append(Olla_result)
    
    # result_list.append(Result_D)
    # result_list.append(Result_A)
    
    # A Static_SINR
    # result_list.append(OLLA_static_sinr)
    # result_list.append(NOLLA_static_sinr_matched)
    # result_list.append(NOLLA_static_sinr_unmatched)
    
    # B with Interference
    # result_list.append(OLLA_with_interference)
    # result_list.append(NOLLA_with_interference)
    
    # C variant_SINR
    # result_list.append(OLLA_variant_sinr)
    # result_list.append(NOLLA_variant_sinr)
    
    # low bler try
    # result_list.append(OLLA_low_bler)
    # result_list.append(NOLLA_low_bler)

    # BLER per UE
    result_list.append(OLLA_per_UE)
    result_list.append(NOLLA_per_UE)
         
    num_time_slots_plot = 1000
    
    save_folder = "figures_for_paper/report_figure/three_comparison/"
    save_figure = True
    plot_bler(result_list, num_time_slots_plot, save_figure, save_folder)
    plot_tputs(result_list, num_time_slots_plot, save_figure, save_folder)
    plot_sinr(result_list, num_time_slots_plot, save_figure, save_folder)
    plot_tputs_cdf(result_list, num_time_slots_plot, save_figure, save_folder)
    plot_bler_cdf(result_list, num_time_slots_plot, save_figure, save_folder)
    plot_sinr_offset(result_list, num_time_slots_plot)
    plot_tputs_subplots(result_list, num_time_slots_plot, save_figure, save_folder)
    plot_bler_per_ue_cdf(result_list, num_time_slots_plot, save_figure, save_folder)
    plt.show()