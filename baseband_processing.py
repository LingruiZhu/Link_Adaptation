import sionna as sn
import numpy as np

from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.utils import BinarySource, ebnodb2no, hard_decisions

def transmit(batch_size:int, num_data_symbols:int, num_bits_per_symbol:int, code_rate:float, constellation:Constellation):
    num_code_bits = int(num_data_symbols*num_bits_per_symbol)
    num_info_bits = int(num_code_bits*code_rate)
    source = BinarySource()
    ldpc_encoder = LDPC5GEncoder(k=num_info_bits, n=num_code_bits)
    mapper = Mapper(constellation=constellation)
    interleaver = RandomInterleaver()

    info_bits = source([batch_size, num_info_bits])
    codewords = ldpc_encoder(info_bits)
    codewords_interleaved = interleaver(codewords)
    symbols = mapper(codewords_interleaved)
    
    return symbols, interleaver


if __name__ == "__main__":
    constellation = Constellation("qam", num_bits_per_symbol=2)
    tx_symbols, intlver = transmit(batch_size=10, num_data_symbols=16, num_bits_per_symbol=2, code_rate=0.5, constellation=constellation)
    print(tx_symbols.shape)
    

