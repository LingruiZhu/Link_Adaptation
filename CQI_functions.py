import numpy as np
import matplotlib.pyplot as plt

def estimate_sinr_from_cqi(cqi, awgn_data):
    REF_BLER_TARGET = 0.1
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]

    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler   = awgn_data['snr_vs_per']

    _, nrof_cqi = awgn_snr_vs_bler.shape

    bler = awgn_snr_vs_bler[:, REF_MCS_INDICES[ cqi ] ]

    if cqi == 0:
        return np.min(awgn_snr_range_dB)
    elif cqi == nrof_cqi - 1:
        return np.max(awgn_snr_range_dB)

    # Find the SNR indices closest to the REF_BLER_TARGET.
    # Estimate the instantaneous SNR by averaging these SNR values.
    # This assumes that the reported CQI actually had a BLER close to REF_BLER_TARGET.
    index1 = np.max(np.argwhere(REF_BLER_TARGET < bler))
    index2 = np.min(np.argwhere(REF_BLER_TARGET > bler))

    estimated_sinr_dB = (awgn_snr_range_dB[index1] + awgn_snr_range_dB[index2]) / 2.0

    return estimated_sinr_dB


def determine_cqi_from_sinr(snr_dB, packet_sizes, awgn_data, cqi_sinr_error = 0.0):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler   = awgn_data['snr_vs_per']

    REF_BLER_TARGET  = 0.1
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]
    nrof_cqi = len( REF_MCS_INDICES )

    # Estimate the BLER for the reference MCSs used to calculate the CQI
    bler_at_snr = determine_bler_at_sinr(snr_dB + cqi_sinr_error, awgn_data)[ REF_MCS_INDICES ]
    
    # Calculate expcted throughput for all valid MCSs
    expected_tputs = np.multiply( ( 1 - bler_at_snr ), np.array( packet_sizes )[ REF_MCS_INDICES ] )
    
    # Ignore any MCSs with BLER less than REF_BLER_TARGET
    expected_tputs[ bler_at_snr > 0.1 ] = 0
    
    # The CQI is the index of the highest-throuput MCS from the reference MCSs
    cqi = 0
    if len( expected_tputs ) > 0:
        cqi = np.argmax( expected_tputs )
    return cqi


def determine_bler_at_sinr(snr_dB, awgn_data):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler   = awgn_data['snr_vs_per']

    _, nrof_mcs = awgn_snr_vs_bler.shape

    bler_at_sinr = np.ndarray((nrof_mcs))

    for i in range(nrof_mcs):
        bler = awgn_snr_vs_bler[:, i]
        
        if snr_dB <= np.min(awgn_snr_range_dB):
            bler_at_sinr[i] = 1.0
        elif snr_dB >= np.max(awgn_snr_range_dB):
            bler_at_sinr[i] = 0.0
        else:
            index1 = np.max(np.argwhere(awgn_snr_range_dB < snr_dB))
            index2 = np.min(np.argwhere(awgn_snr_range_dB > snr_dB))

            bler_at_sinr[i] = ( bler[index1] + bler[index2]) / 2.0

    return bler_at_sinr


def determine_mcs_from_sinr(awgn_data, sinr_dB, bler_target):
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler = awgn_data['snr_vs_bler']

    _, nrof_cqi = awgn_snr_vs_bler.shape

    tbs = [ 20, 20, 40, 64, 84, 104, 124, 148, 168, 148, 188, 232, 272, 316, 356, 400, 408, 472, 536, 600, 660, 724 ]
    
    bler_at_snr = determine_bler_at_sinr(awgn_data, sinr_dB)

    # Find the largest MCS that has BLER less than the BLER target
    # The CQIs are evaluated in decreasing order and first value that predicts a BLER < 0.1
    # is returned.
    largest_mcs = 0
    for i in range(nrof_cqi):
        current_mcs = nrof_cqi - i - 1
        if bler_at_snr[current_mcs] < bler_target:
            largest_mcs = current_mcs
            break 
        else:
            continue

    # Determine the expected tput for all valid MCSs
    best_mcs = 0
    best_expected_tput = 0
    for i in range( largest_mcs ):
        expected_tput = ( 1 - bler_at_snr[ i ] ) * float( tbs[ i ] )
        if expected_tput > best_expected_tput:
            best_expected_tput = expected_tput
            best_mcs = i
    
    return best_mcs


if __name__ == "__main__":
    awgn_data_file = "AWGN_DATASET.npy"
    awgn_data = np.load(awgn_data_file, allow_pickle=True)[ ( ) ]

    snr_vs_bler = awgn_data['snr_vs_per']
    snr_range_dB = awgn_data['snr_range_dB']

    nrof_snr, nrof_rates = snr_vs_bler.shape

    print(snr_vs_bler.shape)
    print(snr_range_dB.shape)

    # Visualize the lookup data
    plt.figure(figsize=[20,5])

    legend = []
    for i in range(nrof_rates):
        plt.semilogy( snr_range_dB, snr_vs_bler[:,i] )
        legend.append('MCS %d'%(i))
        
    plt.legend(legend, ncol=2)
    plt.title('SNR-vs-BLER data for CQI lookup')
    plt.xlabel('Average SNR [dB]')
    plt.ylabel('BLER')
    plt.show()




