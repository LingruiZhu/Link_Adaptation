import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_sinr(file_path:str, num_time_slots:int, trail_index:int=0):
    result_file = h5py.File(file_path, "r")
    sinr_eff_array = np.array(result_file.get("SINR"))[trail_index, -num_time_slots:]
    sinr_feedback_array = np.array(result_file.get("SINR_feedback"))[trail_index, -num_time_slots:]
    sinr_offset_array = np.array(result_file.get("sinr_offset"))[trail_index, -num_time_slots:]
    ACK_NACK_array = np.array(result_file.get("ACK"))[trail_index, -num_time_slots:]
    sinr_real_array = np.array(result_file.get("real_SINR"))[trail_index, -num_time_slots:]
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(sinr_eff_array, "r-x", label="SINR_eff")
    plt.plot(sinr_feedback_array, "b-x", label="SINR_feedback")
    plt.plot(sinr_real_array, "g-o", label="real SINR")
    plt.plot()
    plt.xlabel("Time Slots")
    plt.ylabel("SINR_feedback")
    plt.legend()
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(sinr_offset_array, "k-s", label="SINR_offset")
    plt.plot(ACK_NACK_array, "g-s", label="ACK")
    plt.ylabel("SINR_offset")
    plt.xlabel("Time Slots")
    plt.legend()
    plt.grid()
    
    fig, ax = plt.subplots(figsize=(5,1.5))
    ax.plot(sinr_eff_array, "r-x")
    ax.set_xlabel("Time Slots")
    ax.set_ylabel(r"OLLA $\hat{\gamma}_{eff}$", color="red")
    ax.grid()
    
    ax2 = ax.twinx()
    ax2.plot(ACK_NACK_array, "k-*")
    ax2.set_ylabel("ACK/NACK", color="black")
    plt.show()
    
    fig.savefig("figures_for_paper/SINR_ACK.pdf", format="pdf", bbox_inches="tight")
    

if __name__ == "__main__":
    # plot_sinr(file_path="experiment_simulation_quantized/results_v2/OLLA_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_False_simulation_1220.h5", num_time_slots=1000)
    plot_sinr(file_path="experiment_simulation_quantized/results_nolla_parameter/NOLLA_alpha_down_0.80.h5", num_time_slots=100, trail_index=5)
    plt.show()
    
    
    

    