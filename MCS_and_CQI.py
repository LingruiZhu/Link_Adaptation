class ModulationCodingScheme:
    def __init__(self, coding_scheme:str="LDPC", code_rate:list=None, modulation_order:list=None, mcs_index:list=None) -> None:
        self.coding_scheme = coding_scheme
        self.__code_rate = code_rate
        self.__modulation_order = modulation_order
        self.__mcs_index = mcs_index
        self.__sort_out_code_rate()
    

    def __sort_out_code_rate(self):
        indices = [i for i in range(len(self.__code_rate)) if self.__code_rate[i] > 1/5 and self.__code_rate[i] < 11/12]
        self.code_rate = self.__take_elements(self.__code_rate, indices)
        self.modulation_order = self.__take_elements(self.__modulation_order, indices)
        self.mcs_index = self.__take_elements(self.__mcs_index, indices)
    

    def __take_elements(self, old_list, index_list):
        new_list = [old_list[i] for i in index_list ]
        return new_list


class ChannelQualityIndex:
    def __init__(self, cqi_list, sinr_list) -> None:
        self.cqi_list = cqi_list
        self.sinr_list = sinr_list


def main():
    pass


if __name__ == "__main__":
    main
