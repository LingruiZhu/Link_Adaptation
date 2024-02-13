import numpy as np


class ExperienceBuffer:
    def __init__(self, length:int) -> None:
        self.state:list = list()
        self.action:list = list()
        self.reward:list = list()
        self.following_state:list = list()
        self.max_length = length
        
        
    def add_experience(self, state, action, reward, following_state):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.following_state.append(following_state)
        
        if len(self.state) > self.max_length:
            self.state.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.following_state.pop(0)
    
    
    def get_buffer_length(self):
        return len(self.state)
    
    
    def sample(self, batch_size):
        sample_ids = np.random.choice(len(self.state), size=batch_size)
        mini_batch = list()        
        for i in sample_ids:
            mini_batch.append([self.state[i], self.action[i], self.reward[i], self.following_state[i]])
        return mini_batch
        
    