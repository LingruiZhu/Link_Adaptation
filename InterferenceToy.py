import numpy as np


class InterferenceToy:
    def __init__(self, tx_power, h_factor, interval:int=1, period:int=1, starting_point:int=0, area:list=None) -> None:
        self.tx_power = tx_power
        self.h_factor = h_factor
        self.interval = interval
        self.period = period
        self.starting_point = starting_point
        self.time_instant = 0
        if area is not None:
            self.left_point = area[0]
            self.rignt_point = area[1]
            self.bottom_point = area[2]
            self.top_point = area[3]
    

    def get_interference_time(self):
        interference = 0
        index = self.time_instant % self.period
        if self.time_instant % (self.interval+self.period) < (self.starting_point + self.period):
            interference = self.h_factor * self.tx_power
        self.time_instant += 1
        return interference
    
    
    def is_in_area(self, location):
        pos_x, pos_y = location[0], location[1]
        add_interference = False
        if self.left_point <= pos_x <= self.rignt_point and self.bottom_point <= pos_y <= self.top_point:
            add_interference = True
        return add_interference


    def get_interference_space(self, location):
        interference = 0
        add_interference = self.is_in_area(location)
        if add_interference:
            interference = self.h_factor * self.tx_power
        return interference



if __name__ == "__main__":
    inter_toy = InterferenceToy(1, h_factor=0.01, interval=3, period=2, starting_point=0)
    inter_list = list()
    for i in range(20):
        inter_list.append(inter_toy.get_interference())
    print(inter_list)
