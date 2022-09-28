import math
import numpy as np

class ModulationCodingScheme:
    def __init__(self, coding_scheme:str="LDPC", code_rate:list=None, num_bits_per_symbol:list=None, mcs_index:list=None, 
                tbs:list=None) -> None:
        self.coding_scheme = coding_scheme
        self.__code_rate = code_rate
        self.__num_bits_per_symbol = num_bits_per_symbol
        self.__mcs_index = mcs_index
        self.__tbs = tbs
        self.__sort_out_code_rate()
    

    def __sort_out_code_rate(self):
        indices = [i for i in range(len(self.__code_rate)) if self.__code_rate[i] > 1/5 and self.__code_rate[i] < 11/12]
        self.code_rate = self.__take_elements(self.__code_rate, indices)
        self.number_bits_per_symbol = self.__take_elements(self.__num_bits_per_symbol, indices)
        self.mcs_index = self.__take_elements(self.__mcs_index, indices)
        self.tbs = self.__take_elements(self.__tbs, indices)
    

    def __take_elements(self, old_list, index_list):
        new_list = [old_list[i] for i in index_list]
        return new_list


class ChannelQualityIndex:
    def __init__(self, cqi_list, sinr_list) -> None:
        self.cqi_list = cqi_list
        self.sinr_list = sinr_list
    

    def decide_cqi_from_sinr(self, sinr):
        sinr_array = np.array(self.sinr_list)
        idx = (np.abs(sinr_array - sinr)).argmin()
        cqi = self.cqi_list[idx]
        return cqi
    

    def estimate_sinr_from_cqi(self, cqi):
        idx = self.cqi_list.index(cqi)
        sinr = self.sinr_list[idx]
        return sinr


def get_MCS():
    # from 5.1.3.1-3 MCS index table 3 for PDSCH
    # TODO: also to update the other two tables
    num_bits_per_symbol = [2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2,
                        2, 2, 2, 2, 2,
                        4, 4, 4, 4, 4,
                        4, 6, 6, 6, 6,
                        6, 6, 6, 6]
    info_bits_length = [20, 40, 50, 64, 78,
                        99, 120, 157, 193, 251,
                        308, 379, 449, 526, 602,
                        340, 378, 434, 490, 553,
                        616, 438, 466, 517, 567,
                        616, 666, 719, 772] 
    code_rate = [x/1024 for x in info_bits_length]
    mcs_index = [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28]
    num_data_symbols = 720
    tran_block_size = [math.floor(num_data_symbols*num_bits*c_rate) for num_bits, c_rate in zip(num_bits_per_symbol, code_rate)]
    mcs = ModulationCodingScheme(code_rate=code_rate, num_bits_per_symbol=num_bits_per_symbol, mcs_index=mcs_index, \
                                 tbs=tran_block_size)
    return mcs


def get_CQI():
    sinr = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7,
            14.1, 16.3, 18.7, 21.0, 22.7]
    cqi_code = [1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10,
                11, 12, 13, 14, 15]
    cqi = ChannelQualityIndex(cqi_code, sinr)
    return cqi


def main():
    cqi_table = get_CQI()
    cqi_code = cqi_table.decide_cqi_from_sinr(0)
    print(cqi_code)
    est_sinr = cqi_table.estimate_sinr_from_cqi(cqi_code)
    print(est_sinr)


if __name__ == "__main__":
    main()
