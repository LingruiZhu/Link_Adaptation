import numpy as np

class DQNConfig:
    def __init__(self, gamma:float, epsilon_decay:float) -> None:
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        