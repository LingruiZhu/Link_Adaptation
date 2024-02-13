import numpy as np
from experiment_simulation_quantized.simulation_template import OLLA_simulation
from la_agent.olla_agent_non_linear import NonLinear_OLLA
from la_agent.olla_agent import OuterLoopLinkAdaptation


def NOLLA_parameter_simulation(alpha_down):
    # simulation tuning
    num_trails = 20
    num_time_slots = 500
    
    # cqi configuration
    feedback_interval = 5
    num_quant_bits = 4
    cqi_delay = 0
    add_interference = False
    variant_sinr = False
    
    # LA agent
    Nolla_agent = NonLinear_OLLA(bler_target=0.1,
                                 data_file="/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                 olla_step_size=0.1,
                                 alpha_down=alpha_down)
    file_name = "NOLLA_alpha_down_" + str(alpha_down)
    # OLLA_simulation(Nolla_agent, num_trails, num_time_slots, feedback_interval, num_quant_bits, cqi_delay=cqi_delay, file_name=file_name, add_interference=add_interference, variant_sinr=variant_sinr)
    olla_agent = OuterLoopLinkAdaptation(bler_target=0.1, 
                                data_file="/home/zhu/Codes/link_adaptation/BLER_LUT_data_simulation/table3_LUT_AWGN_simulation.h5",
                                olla_step_size=0.1)
    olla_file_name = "OLLA_bench_mark"
    OLLA_simulation(olla_agent, num_trails, num_time_slots, feedback_interval, num_quant_bits, cqi_delay=cqi_delay, file_name=olla_file_name, add_interference=add_interference, variant_sinr=variant_sinr)
    
    
def Nolla_parameter_test():
    for i in np.arange(0.6, 1, 0.05):
        NOLLA_parameter_simulation(alpha_down=i)



if __name__ == "__main__":
    # Nolla_parameter_test()
    NOLLA_parameter_simulation(alpha_down=10)