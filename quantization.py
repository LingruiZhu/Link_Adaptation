import numpy as np

class UniformQuantizer:
    def __init__(self, min:float, max:float, num_bits:int) -> None:
        self.min_value = min
        self.max_value = max
        self.num_bits = num_bits
        self.num_levels = 2**(num_bits)
        self.quant_resolution = (max - min) / self.num_levels
    
    
    def __call__(self, x):
        quant_index = np.math.floor((x - self.min_value) / self.quant_resolution)
        if (x - self.min_value) % self.quant_resolution == 0:
            quant_index -= 1
        x_quant = (quant_index + 0.5) * self.quant_resolution + self.min_value
        return x_quant


if __name__ == "__main__":
    a = 5
    quantizer = UniformQuantizer(min=0, max=10, num_bits=2)
    a_quant = quantizer(a)
    print(a_quant)

