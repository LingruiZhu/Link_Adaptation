import sionna as sn
import numpy as np

from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.utils import BinarySource, ebnodb2no, hard_decisions
from sionna.ofdm import ResourceGrid, LMMSEEqualizer, ResourceGridMapper
from sionna.ofdm.channel_estimation import LSChannelEstimator
from sionna.mimo import StreamManagement

from Simulation_Parameters import Simulation_Parameter


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
    

    def update_mcs(self, modulation_order:int, code_rate:float):
        self.num_bits_per_symbol = modulation_order
        self.code_rate = code_rate
        self.num_code_bits = int(self.num_data_symbols * self.num_bits_per_symbol)
        self.num_info_bits = int(self.num_code_bits * self.code_rate)

        self.__initialize_transmitter()
        self.__initialize_receiver


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
                                    min_speed=self.ue_speed)
        self.channel = sn.channel.OFDMChannel(CDL, self.resource_grid, add_awgn=True, normalize_channel=True, return_channel=True)


    def __initialize_receiver(self):
        self.ls_estimator = LSChannelEstimator(self.resource_grid, interpolation_type="nn")
        self.lmmse_equalizer = LMMSEEqualizer(self.resource_grid, self.stream_management)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.deinterleaver = Deinterleaver(self.interleaver)
        self.ldpc_decoder = LDPC5GDecoder(self.ldpc_encoder, hard_out=True)


    def snr_to_noise_variance(self, ebno_dB):
        no = sn.utils.ebnodb2no(ebno_dB,
                        num_bits_per_symbol=self.num_bits_per_symbol,
                        coderate=self.code_rate,
                        resource_grid=self.resource_grid)
        return no


    def transmit(self):
        info_bits = self.source([self.batch_size, self.num_UE, self.resource_grid.num_streams_per_tx, self.num_info_bits])
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
        decoded_bits = self.ldpc_decoder(llr)
        return decoded_bits


    def run(self, ebno_db):
        # set up simulation parameters
        tx_symbols, info_bits = self.transmit()
        rx_symbols, channel_freq = self.go_through_channel(tx_symbols, ebno_db)
        decoded_bits = self.receive(rx_symbols, ebno_db=10)
        bler = sn.utils.compute_ber(info_bits, decoded_bits)
        return bler


    

