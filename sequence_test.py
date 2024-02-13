import random
import numpy as np

import matplotlib.pyplot as plt


def generate_sequence(num_samples, error_rate):
    seq = list()
    for i in range(num_samples):
        if np.random.uniform() < error_rate:
            seq.append(0)
        else:
            seq.append(1)
    return seq


def count_ma_mn(samples, max_number):
    ma_list = list()
    mn_list = list()
    ma = 0
    mn = 0
    
    for element in samples:
        if element:
            ma = ma+1
            mn = 0
        else:
            mn = mn + 1
            ma = 0
        ma = max_number if ma > max_number else ma
        mn = max_number if mn > max_number else mn        
        ma_list.append(ma)
        mn_list.append(mn)
    return ma_list, mn_list
    


def count_consecutive_samples(samples, max_number):
    zeros_counts_list = list()
    ones_counts_list = list()
    zeros_count = 0
    ones_count = 0
    previous_sample = None
    
    for sample in samples:
        if previous_sample == 0:
            if sample == 0:
                zeros_count += 1
            elif sample == 1:
                zeros_counts_list.append(zeros_count)
                zeros_count=0
                ones_count=1
        elif previous_sample == 1:
            if sample == 1:
                ones_count += 1
            elif sample == 0:
                ones_counts_list.append(ones_count)
                ones_count = 0
                zeros_count = 1
        elif previous_sample == None:
            if sample == 1:
                ones_count += 1
            elif sample == 0:
                zeros_count += 1
        previous_sample = sample
    
    if ones_count:
        ones_counts_list.append(ones_count)
    if zeros_count:
        zeros_counts_list.append(zeros_count)
    
    cutted_ones_counts_list = list()
    for element in ones_counts_list:
        if element > max_number:
            element = max_number
        cutted_ones_counts_list.append(element)
    
    avg_zeros_number = np.mean(zeros_counts_list)
    avg_ones_number = np.mean(cutted_ones_counts_list)
    return cutted_ones_counts_list, zeros_counts_list, avg_ones_number, avg_zeros_number


if __name__ == "__main__":
    sequence = generate_sequence(200000, 0.5)
    ma, mn = count_ma_mn(sequence, 5)
    print(sequence)
    
    count_list = list()
    prob_list = list()
    for i in range(6):
        count_list.append(ma.count(i))
        prob_list.append(ma.count(i)/(len(sequence)))
    # print(ma)
    # print(mn)
    print(count_list)
    print(prob_list)
    
    # one_seq, zero_seq, ave_one, avg_zero = count_consecutive_samples(sequence, 5)

    # seq_count_list = list()
    # seq_prob_count_list = list()    
    # for i in range(6):
    #     seq_count_list.append(one_seq.count(i))
    #     seq_prob_count_list.append(one_seq.count(i)/len(one_seq))
    # print(seq_count_list)
    # print(seq_prob_count_list)

    