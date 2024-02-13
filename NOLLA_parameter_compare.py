import numpy as np


def compare_nolla_parameter(max_ack:int, max_nack:int, alpha_up:float, alpha_down:float, bler:float=0.5):
    # compare the parameter pair: alpha_up for max_nack, alpha_down for max_ack
    m_ack_prob_list = list()
    alpha_down_list = list()
    for i in range(max_ack):
        prob = bler * ((1-bler)**i)
        m_ack_prob_list.append(prob)
        if i == 0:
            alpha_down_list.append(0)
        else:
            alpha_down_list.append(alpha_down**i)
    last_m_ack_prob = 1 - np.sum(m_ack_prob_list)
    m_ack_prob_list.append(last_m_ack_prob)
    alpha_down_list.append(alpha_down**max_ack)
    m_ack_prob_list_norm = [i / (1 - m_ack_prob_list[0]) for i in m_ack_prob_list]
    
    m_nack_prob_list = list()
    alpha_up_list = list()
    for i in range(max_nack):
        prob = (1-bler) * (bler**i)
        m_nack_prob_list.append(prob)
        if i == 0:
            alpha_up_list.append(0)
        else:
            alpha_up_list.append(alpha_up**i)
    last_m_nack_prob = 1 - np.sum(m_nack_prob_list)
    m_nack_prob_list.append(last_m_nack_prob)
    alpha_up_list.append(alpha_up**max_nack)
    m_nack_prob_list_norm = [i / (1 - m_nack_prob_list[0]) for i in m_nack_prob_list]
    
    mean_alpha_up_to_m_nack = np.sum(np.array(m_nack_prob_list_norm)*np.array(alpha_up_list))
    mean_alpha_down_to_m_ack = np.sum(np.array(m_ack_prob_list_norm)*np.array(alpha_down_list))
    ratio = mean_alpha_up_to_m_nack/mean_alpha_down_to_m_ack
    
    return mean_alpha_up_to_m_nack, mean_alpha_down_to_m_ack, ratio


def parameter_test():
    max_ack = 10
    max_nack = 10
    alpha_up = 0.88
    alpha_down = 0.72
    
    mean_up_nack, mean_down_ack, ratio = compare_nolla_parameter(max_ack, max_nack, alpha_up, alpha_down)
    print(mean_up_nack, mean_down_ack, ratio)
    

if __name__ == "__main__":
    parameter_test()