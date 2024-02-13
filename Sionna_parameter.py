from sionna.ofdm import ResourceGrid

def get_default_rg():
    """this funtion returns a default resourse grid with pilot, data and guard carriers
    """
    resouce_grid = ResourceGrid(num_ofdm_symbols=14,
                                fft_size=72,
                                subcarrier_spacing=30e3,
                                num_tx=1,
                                num_streams_per_tx=1,
                                cyclic_prefix_length=6,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=[2, 11],
                                num_guard_carriers = [5, 6],
                                dc_null=True
                                )
    print(resouce_grid.num_data_symbols)
    return resouce_grid