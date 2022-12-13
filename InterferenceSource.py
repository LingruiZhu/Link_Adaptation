import numpy as np


class InterfernceSource:
    def __init__(self, position:tuple, tx_power:float, interval:int, rice_factor:float, period:int = 1, starting_point:int = 0) -> None:
        self.x_pos = position[0]
        self.y_pos = position[1]
        self.tx_power = tx_power
        self.transmit_interval = interval
        self.transmit_period = period
        self.starting_point = starting_point
        self.rice_factor = rice_factor
    

    def calculate_los_channel(self):
        """in this functin, LOS (Line of Sight) channel coefficient is calculated.
        Now, assume single anttena. Multi-anttenas will be added in the future if necessary.
        """
        ue_distance = np.sqrt(self.x_pos**2 + self.y_pos**2)
        aod = np.arcsin(np.abs(self.y_pos)/ue_distance)
        aod_gain = 1
        k0 = 1e-6
        channelL_los = np.sqrt()


    def calculate_interference(self):
        pass

    
