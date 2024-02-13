import numpy as np
import h5py

class SINRSequence:
    def __init__(self, file_path:str) -> None:
        with h5py.File(file_path, "r") as f:
            self.real_SINR_list = f["SINR_real"][:]
            self.pred_SINR_list = f["SINR_prediction"][:]
        self.time_index = 0
    
    
    def get_SINR_value(self):
        real_SINR_value = self.real_SINR_list[self.time_index]
        pred_SINR_value = self.pred_SINR_list[self.time_index]
        self.time_index = self.time_index + 1
        return real_SINR_value, pred_SINR_value


def main():
    # TODO: 1) test SINRSequence. 2) Check the keyword in h5 File. 
    file_path = "SINR_sequence/embedding_loss_random_input_40_latent_20_num_embeddings_64_init_random_RMSprop_0.h5"
    sinr_seq = SINRSequence(file_path)
    print(sinr_seq.real_SINR_list[:5])
    print(sinr_seq.pred_SINR_list[:5])
    for i in range(5):
        print(sinr_seq.get_SINR_value())
    


if __name__ == '__main__':
    main()
    