import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_loss():
    file_path = "experiment_simulation_quantized/results_v2/OLLA_DQN_v2_ack_weight_0_num_time_slots_4_num_trails_20_cqi_interval_5_quant_bits_4_delay_0_add_interference_True_reward_and_more_actions.h5"
    data_file = h5py.File(file_path, "r")
    loss_array = np.array(data_file.get("loss_history"))
    mean_loss = np.mean(loss_array, axis=0)
    individual_loss = loss_array[0,:]
    
    plt.figure()
    plt.plot(mean_loss)
    plt.grid()
    
    plt.figure()
    plt.plot(individual_loss)
    plt.grid()
    
    plt.show()    

if __name__ == "__main__":
    plot_loss()
    