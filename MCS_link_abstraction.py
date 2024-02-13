import numpy as np
import h5py
from sionna.utils import sim_ber 
from Simulation_Parameters import Simulation_Parameter, get_default_rg, Channel_Model
from LinkSimulation import Link_Simulation
from MCS_and_CQI import get_CQI, get_MCS


def get_link_abstraction_file_name(mcs_index):
    file_name = "BLER_link_abstraction/"+"BLER_MCS_" + str(mcs_index) + ".h5"
    return file_name


def e2e_simulation(code_rate, num_bits_per_symbol, mcs_index):
    # parameter configuration
    resourse_grid = get_default_rg()
    carrier_frequency = 2.6e9
    ue_speed = 0
    delay_spread = 100-9
    channel_model = Channel_Model.AWGN
    
    sim_parameters = Simulation_Parameter(resource_grid=resourse_grid,
                                          num_bits_per_symbol=num_bits_per_symbol,
                                          code_rate=code_rate,
                                          carrier_frequency=carrier_frequency, 
                                          ue_speed=ue_speed,
                                          delay_spread=delay_spread,
                                          channel_model=channel_model)
    link_simulation = Link_Simulation(sim_paras=sim_parameters)
    eb_no_list = np.arange(4,15,0.1)
    batch_size = 1000
    ber, bler = sim_ber(link_simulation, 
                        ebno_dbs=eb_no_list,
                        batch_size = batch_size,
                        num_target_block_errors=50,
                        max_mc_iter=50,
                        verbose=False,
                        early_stop=True)
    
    result_file_name = get_link_abstraction_file_name(mcs_index)
    result_file = h5py.File(result_file_name, "w")
    result_file.create_dataset("BLER", data=bler.numpy())
    result_file.create_dataset("BER", data=ber.numpy())
    result_file.create_dataset("eb_no_db", data=np.array(eb_no_list))
    result_file.close
    

def MCS_bler_simulation():
    MCS_set = get_MCS()
    code_rate_list = MCS_set.code_rate
    num_bits_per_symbol_list = MCS_set.number_bits_per_symbol
    mcs_indicies = MCS_set.mcs_index
    
    for num_bits_per_symbol, code_rate, mcs_index in zip(num_bits_per_symbol_list, code_rate_list, mcs_indicies):
        e2e_simulation(code_rate=code_rate,
                       num_bits_per_symbol=num_bits_per_symbol,
                       mcs_index=mcs_index)


if __name__ == "__main__":
    MCS_bler_simulation()
    