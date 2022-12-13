import numpy as np
from sionna.channel import RayleighBlockFading
from sionna.channel.tr38901 import TDL

def generate_sionna_channel():
    tdl = TDL(model = "A",
               delay_spread = 300e-9,
               carrier_frequency = 3.5e9,
               min_speed = 20,
               max_speed = 30)
    h, delays = tdl(batch_size=1, num_time_steps=1000, sampling_frequency=4e6)
    h_2d = np.squeeze(np.absolute(h))
    return h, delays


def generate_block_fading(batch_size, time_samples:int):
    block_fading_channel = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1)
    channel_coefficients, delays = block_fading_channel(batch_size=batch_size, num_time_steps=time_samples)
    return channel_coefficients, delays


def generate_rayleigh_coefficients(std:float):
    h_real = np.random.normal(loc=0)

if __name__ == "__main__":
    h, delays = generate_sionna_channel()
