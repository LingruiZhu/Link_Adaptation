import sionna as sn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.utils import BinarySource, ebnodb2no, hard_decisions
from sionna.ofdm import ResourceGrid, LMMSEEqualizer, ResourceGridMapper
from sionna.ofdm.channel_estimation import LSChannelEstimator
from sionna.mimo import StreamManagement
from sionna.channel import ApplyOFDMChannel, AWGN
from sionna.ofdm.pilot_pattern import PilotPattern

from Simulation_Parameters import Simulation_Parameter, Channel_Model
from channel_data.read_channel_data import read_channel_data


class Link_Simulation():
    def __init__(self, sim_paras:Simulation_Parameter):
        # Set up parameters
        self.resource_grid = sim_paras.resource_grid
        self.num_data_symbols = self.resource_grid.num_data_symbols
        self.num_bits_per_symbol = sim_paras.num_bits_per_symbol
        self.batch_size = sim_paras.batch_size
        self.code_rate = sim_paras.code_rate
        self.num_code_bits = int(self.num_data_symbols * self.num_bits_per_symbol)
        self.num_info_bits = int(self.num_code_bits * self.code_rate)
        self.ue_speed = sim_paras.ue_speed
        self.carrier_frequency = sim_paras.carrier_frequency
        self.delay_spread = sim_paras.delay_spread
        self.channel_model = sim_paras.channel_model

        # parameters for MIMO, but here everything is single. 
        self.num_UE = 1                     # single user and base station
        self.num_BS = 1
        self.num_UE_ANT = 1                 # both with single antenna
        self.num_BS_ANT = 1
        self.rx_tx_association = np.array([[1]])
        self.stream_management = StreamManagement(self.rx_tx_association, num_streams_per_tx=self.num_UE_ANT)
            
        self.__initialize_transmitter()     # Set up transmitter
        self.__initialize_channel()         # Set up channel
        self.__initialize_receiver()        # Set up receiver
    

    def __initialize_transmitter(self):
        self.source = BinarySource()
        self.ldpc_encoder = LDPC5GEncoder(k=self.num_info_bits, n=self.num_code_bits)
        self.constellation = Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.resource_grid_mapper = ResourceGridMapper(self.resource_grid)
        self.interleaver = RandomInterleaver()


    def __initialize_channel(self):
        # ToDO: possibly need to have a another function to re-seed the channel
        UE_Array = sn.channel.tr38901.Antenna( polarization="single",
                                            polarization_type="V",
                                            antenna_pattern="38.901",
                                            carrier_frequency=self.carrier_frequency)
        BS_Array = sn.channel.tr38901.AntennaArray(num_rows=1,
                                            num_cols=1,
                                            polarization="single",
                                            polarization_type="V",
                                            antenna_pattern="38.901", # Try 'omni'
                                            carrier_frequency=self.carrier_frequency)
        direction = "downlink"
        CDL_model = "C"
        CDL = sn.channel.tr38901.CDL(CDL_model,
                                    delay_spread=self.delay_spread,
                                    carrier_frequency=self.carrier_frequency,
                                    ut_array=UE_Array,
                                    bs_array=BS_Array,
                                    direction=direction,
                                    min_speed=self.ue_speed,
                                    max_speed=self.ue_speed)
        if self.channel_model == Channel_Model.CDL:
            self.channel = sn.channel.OFDMChannel(CDL, self.resource_grid, add_awgn=True, normalize_channel=True, return_channel=True)
        elif self.channel_model == Channel_Model.AWGN:
            self.channel = AWGN()


    def __initialize_receiver(self):
        self.ls_estimator = LSChannelEstimator(self.resource_grid, interpolation_type="nn")
        self.lmmse_equalizer = LMMSEEqualizer(self.resource_grid, self.stream_management)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.deinterleaver = Deinterleaver(self.interleaver)
        self.ldpc_decoder = LDPC5GDecoder(self.ldpc_encoder, hard_out=True)


    def update_mcs(self, modulation_order:int, code_rate:float):
        self.num_bits_per_symbol = modulation_order
        self.code_rate = code_rate
        self.num_code_bits = int(self.num_data_symbols * self.num_bits_per_symbol)
        self.num_info_bits = int(self.num_code_bits * self.code_rate)

        self.__initialize_transmitter()
        self.__initialize_receiver


    def snr_to_noise_variance(self, ebno_dB):
        no = sn.utils.ebnodb2no(ebno_dB,
                        num_bits_per_symbol=self.num_bits_per_symbol,
                        coderate=self.code_rate,
                        resource_grid=self.resource_grid)
        return no


    def transmit(self, batch_size:int=None):
        if batch_size == None:
            batch_size = self.batch_size
        info_bits = self.source([batch_size, self.num_UE, self.resource_grid.num_streams_per_tx, self.num_info_bits])
        codewords = self.ldpc_encoder(info_bits)
        codewords_interleaved = self.interleaver(codewords)
        symbols = self.mapper(codewords_interleaved)
        symbols_rg = self.resource_grid_mapper(symbols)
        return symbols_rg, info_bits
    

    def go_through_channel(self, tx_symbols, ebno_db):
        no = self.snr_to_noise_variance(ebno_db)
        rx_symbols, h_freq = self.channel([tx_symbols, no])
        return rx_symbols, h_freq
        

    def receive(self, rx_symbols, ebno_db):
        no = self.snr_to_noise_variance(ebno_db)
        channel_estimation, error_variance = self.ls_estimator([rx_symbols, no])
        equalized_symbols, no_eff = self.lmmse_equalizer([rx_symbols, channel_estimation, error_variance, no])
        llr = self.demapper([equalized_symbols, no_eff])
        llr_deintlv = self.deinterleaver(llr)
        decoded_bits = self.ldpc_decoder(llr_deintlv)
        return decoded_bits


    def run(self, ebno_db):
        # set up simulation parameters
        tx_symbols, info_bits = self.transmit()
        rx_symbols, channel_freq = self.go_through_channel(tx_symbols, ebno_db)
        decoded_bits = self.receive(rx_symbols, ebno_db=10)
        ber = sn.utils.compute_ber(info_bits, decoded_bits)
        bler = sn.utils.compute_bler(info_bits, decoded_bits)
        return ber, bler

    
    def go_through_channel_single_PRB(self, channel_matrix, tx_symbols, ebno_db):
        no = self.snr_to_noise_variance(ebno_db) 
        applied_channel = ApplyOFDMChannel()
        channel_matrix = channel_matrix.astype("complex64")
        rx_symbols = applied_channel([tx_symbols, channel_matrix, no])
        return rx_symbols
    

    def simulate_single_PRB(self, channel_matrix, ebno_db):
        tx_symbols, info_bits = self.transmit(batch_size=1)
        rx_symbols = self.go_through_channel_single_PRB(channel_matrix, tx_symbols, ebno_db)
        decoded_bits = self.receive(rx_symbols, ebno_db)
        ber = sn.utils.compute_ber(info_bits, decoded_bits)
        bler = sn.utils.compute_bler(info_bits, decoded_bits) 
        ack = not bool(bler)
        tsb_size = int(tf.size(info_bits))
        return ber, bler, ack, tsb_size # TODO:here to add to calculate SINR and then CQI


def test_a():
    """Just a function to test Link Adaptation class """

    resouce_grid = ResourceGrid(num_ofdm_symbols=14,
                                fft_size=76,
                                subcarrier_spacing=30e3,
                                num_tx=1,
                                num_streams_per_tx=1,
                                cyclic_prefix_length=6,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=[2, 11])
    batch_size = 200
    num_bits_per_symbol = 4
    code_rate = 0.5
    carrier_frequency = 2.6e9
    ue_speed = 0
    delay_spread = 100e-9#
    sim_para_default = Simulation_Parameter(resouce_grid, batch_size, num_bits_per_symbol, code_rate, carrier_frequency, ue_speed, delay_spread)
    link_simulator = Link_Simulation(sim_paras=sim_para_default)

    #TODO: here need a function  which reads data from .npy file.
    data_file = "channel_data/CDL_channel.h5"
    channel_data = read_channel_data(data_file, start_point=0, num_PRB=10)
    channel_sample = channel_data[0:1,:]
    ber, bler, ack, tsb_size = link_simulator.simulate_single_PRB(channel_sample, ebno_db=10)
    
    snrs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bler_list = list()
    ber_list = list()
    for snr in snrs:
        ber, bler = link_simulator.run(snr)
        ber_list.append(ber)
        bler_list.append(bler)

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(snrs, ber_list)
    ax1.set_yscale("log")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("BER")

    ax2.plot(snrs, bler_list)
    ax2.set_yscale("log")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("BLER")
    plt.show()


if __name__ == "__main__":
    test_a()    
    
