import sys
sys.path.append("/home/zhu/Codes/link_adaptation")

from MCS_and_CQI import ChannelQualityIndex, ModulationCodingScheme

def get_mcs_scenario_a():
    code_rate = [0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]
    bits_per_symbol = [2, 4, 6]
