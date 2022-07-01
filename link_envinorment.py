import numpy as np
import gym

from baseband_processing import Link_Simulation
from Simulation_Parameters import Simulation_Parameter, get_default_parameters

class LinkEnvironment:
    def __init__(self,) -> None:
        # define link simulation 
        default_sim_paras = get_default_parameters()
        self.link_simulation = Link_Simulation(default_sim_paras)
        
        
    def step(self, modulation_order, code_rate):
        self.link_simulation.update_mcs(modulation_order=)
        

