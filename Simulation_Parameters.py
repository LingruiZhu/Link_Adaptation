from dataclasses import dataclass
from sionna.ofdm import ResourceGrid

@dataclass
class Simulation_Parameter:
    resource_grid:ResourceGrid
    num_batches:int
    num_bits_per_symbol:int
    code_rate:float
    carrier_frequency:float
    ue_speed:float
    delay_spread:float