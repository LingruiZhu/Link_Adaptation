import numpy as np


class RayleighFadingChannel:
    def __init__(self, ue_speed:float) -> None:
        self.relative_spped = ue_speed
        self.