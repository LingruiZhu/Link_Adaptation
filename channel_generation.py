import numpy as np
from enum import Enum
import sionna as sn
import h5py

from sionna.ofdm import ResourceGrid
from sionna.channel import RayleighBlockFading, GenerateOFDMChannel

from Simulation_Parameters import Simulation_Parameter


class Channel_type(Enum):
    RayleighBlockFading = 1
    tri38901_CDL = 2


def initialize_channel_model(ch_type:Channel_type, sim_paras:Simulation_Parameter):
    # get parameter
    carrier_frequency = sim_paras.carrier_frequency
    delay_spread = sim_paras.delay_spread
    ue_speed = sim_paras.ue_speed

    # define diferent channel model
    if ch_type == Channel_type.RayleighBlockFading:
        channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1)
    elif ch_type == Channel_type.tri38901_CDL:
        UE_Array = sn.channel.tr38901.Antenna( polarization="single",
                                            polarization_type="V",
                                            antenna_pattern="38.901",
                                            carrier_frequency=carrier_frequency)
        BS_Array = sn.channel.tr38901.AntennaArray(num_rows=1,
                                            num_cols=1,
                                            polarization="single",
                                            polarization_type="V",
                                            antenna_pattern="38.901", # Try 'omni'
                                            carrier_frequency=carrier_frequency)
        direction = "downlink"
        CDL_model = "C"
        channel_model = sn.channel.tr38901.CDL(CDL_model,
                                    delay_spread=delay_spread,
                                    carrier_frequency=carrier_frequency,
                                    ut_array=UE_Array,
                                    bs_array=BS_Array,
                                    direction=direction,
                                    min_speed= ue_speed-2,
                                    max_speed= ue_speed+2)
        return channel_model


def clip_resource_block(h_freq, num_ofdm_symbols, num_resource_blocks):
    num_subcarriers = h_freq.shape[-1]
    h_freq_rb_scale = np.zeros([num_resource_blocks, 1,1,1,1, num_ofdm_symbols, num_subcarriers], dtype=complex)
    for rb_index in range(num_resource_blocks):
        h_freq_rb_scale[rb_index,:,:,:,:,:,:] = h_freq[:,:,:,:,:,rb_index*num_ofdm_symbols:(rb_index+1)*num_ofdm_symbols,:]
    return h_freq_rb_scale


def save_channel_data(h_freq_rb_scale:np.ndarray, ch_type:Channel_type, sim_paras:Simulation_Parameter, resource_gird:ResourceGrid, save_path=str):
    channel_file = h5py.File(save_path, "w")
    channel_model_name = ch_type.name
    carrier_frequency = sim_paras.carrier_frequency
    ue_speed = sim_paras.ue_speed
    delay_spread = sim_paras.delay_spread

    channel_file.attrs["channel_model"] = channel_model_name
    channel_file.attrs["carrier_frequency"] = carrier_frequency
    channel_file.attrs["ue_spped"] = ue_speed
    channel_file.attrs["delay_spread"] = delay_spread
    channel_file.create_dataset("channel_data", data=h_freq_rb_scale)
    channel_file.close()
    print("channel data has been saved to " + save_path)


def generate_channel(channel_type:Channel_type,sim_paras:Simulation_Parameter, num_ofdm_symbols:int, num_resource_block:int):
    # define resource grid
    resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols*num_resource_block,
                                 fft_size=76,
                                 subcarrier_spacing=30e3,
                                 num_tx=1,
                                 num_streams_per_tx=1,
                                 cyclic_prefix_length=6,
                                 pilot_pattern="kronecker",
                                 pilot_ofdm_symbol_indices=[2, 11])
    channel_model = initialize_channel_model(channel_type, sim_paras)
    generate_channel = GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=True)
    h_freq = generate_channel(batch_size=1)
    h_freq_rb_scale = clip_resource_block(h_freq, num_ofdm_symbols, num_resource_block)
    file_name = input("please enter file name: ")
    file_path = "channel_data/" + file_name + ".h5"
    save_channel_data(h_freq_rb_scale, channel_type, sim_paras, resource_grid, file_path)


if __name__ == "__main__":
    ch_type = Channel_type.tri38901_CDL
    num_ofdm_symbols = 14
    sim_paras = Simulation_Parameter(carrier_frequency=2.6e9,
                                     delay_spread=100e-9,
                                     ue_speed=10)
    generate_channel(ch_type, sim_paras, num_ofdm_symbols, num_resource_block=1000)


    