from dataclasses import dataclass
from sionna.ofdm import ResourceGrid

@dataclass
class Simulation_Parameter:
    resource_grid:ResourceGrid
    batch_size:int
    num_bits_per_symbol:int
    code_rate:float
    carrier_frequency:float
    ue_speed:float
    delay_spread:float


    def set_code_rate(self, code_rate):
        self.code_rate = code_rate


    def set_num_bits_per_symbol(self, num_bits_per_symbol):
        self.num_bits_per_symbol = num_bits_per_symbol



if __name__ == "__main__":
    resouce_grid = ResourceGrid(num_ofdm_symbols=14,
                                fft_size=76,
                                subcarrier_spacing=30e3,
                                num_tx=1,
                                num_streams_per_tx=1,
                                cyclic_prefix_length=6,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=[2, 11])
    batch_size = 100
    num_bits_per_symbol = 4
    code_rate = 0.5
    carrier_frequency = 2.6e9
    ue_speed = 10
    delay_spread = 100e-9#

    sim_paras = Simulation_Parameter(resouce_grid, batch_size, num_bits_per_symbol, code_rate, carrier_frequency, ue_speed, delay_spread)

    print(sim_paras.num_bits_per_symbol)
    print(sim_paras.code_rate)

    new_modulation_order = 8
    new_code_rate = 0.75

    sim_paras.set_code_rate(new_code_rate)
    sim_paras.set_num_bits_per_symbol(new_modulation_order)

    print(sim_paras.num_bits_per_symbol)
    print(sim_paras.code_rate)

