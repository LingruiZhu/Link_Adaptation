import numpy as np
import matplotlib.pyplot as plt

class Moving_Object:
    def __init__(self, position:np.ndarray, motion_matrix:np.ndarray) -> None:
        self.position = position
        self.motion_matrix = motion_matrix
        self.position_history = [position]
    
    def update(self):
        self.position = np.matmul(self.motion_matrix, self.position) + np.random.normal(0, 0.1)
        self.position_history.append(self.position)
    

def plot_movement():
    motion_matrix = np.array([[0.9921, -0.1247], [0.1247, 0.9921]])
    ini_position = np.array([10,0])
    moving_point = Moving_Object(ini_position, motion_matrix)
    num_iterations = 2000

    for i in range(num_iterations):
        moving_point.update()
    
    moving_history = moving_point.position_history
    x_pos_history, y_pos_history = np.array(moving_history)[:,0], np.array(moving_history)[:,1]
    plt.figure()
    plt.plot(x_pos_history, y_pos_history, "b-x")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_movement()

    
