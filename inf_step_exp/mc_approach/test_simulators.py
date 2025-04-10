import sys
import numpy as np
import pandas as pd
import polars as pl
from test_case import Test_Simulator
from crazy_simulation import AMMSimulator

def compare_simulators(params):
    """
    Compare Test_Simulator and AMMSimulator results with the given parameters.
    """
    # Test parameters
    sigma = params['sigma']
    gamma = params['gamma']
    dt = params['dt']
    steps = params['steps']
    seed = params['seed']
    tolerance = params['tolerance']
    
    # Run Test_Simulator
    test_simulator = Test_Simulator(sigma=sigma, gamma=gamma, dt=dt, steps=steps, seed=seed)
    test_df = test_simulator.simulate()
    
    # Run AMMSimulator
    crazy_simulator = AMMSimulator(num_seeds=1, gamma_values=np.array([gamma]), sigma_values=np.array([sigma]), steps=steps, dt=dt)
    crazy_simulator.simulate(output_dir='/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/mc_results')
    crazy_df = pl.read_parquet('/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/mc_results/simulation_results_0.parquet')
    # convert crazy_df to dataframe
    crazy_df = crazy_df.to_pandas()
    print(crazy_df.to_markdown())
    
    # Get the final row from test_df
    test_final = test_df.iloc[-1]
    print(test_final)
    
    # Compare each parameter
    assert np.abs(test_final['price'] - crazy_df['price'].item()) < tolerance, \
        f"Price mismatch: Test={test_final['price']}, AMM={crazy_df['price'].item()}"
    
    assert np.abs(test_final['dis_x'] - crazy_df['x_dis'].item()) < tolerance, \
        f"dis_x mismatch: Test={test_final['dis_x']}, AMM={crazy_df['x_dis']}"
    
    assert np.abs(test_final['dis_y'] - crazy_df['y_dis'].item()) < tolerance, \
        f"dis_y mismatch: Test={test_final['dis_y']}, AMM={crazy_df['y_dis']}"
    
    assert np.abs(test_final['re_in_x'] - crazy_df['x_rinc'].item()) < tolerance, \
        f"re_in_x mismatch: Test={test_final['re_in_x']}, AMM={crazy_df['x_rinc']}"
    
    assert np.abs(test_final['re_in_y'] - crazy_df['y_rinc'].item()) < tolerance, \
        f"re_in_y mismatch: Test={test_final['re_in_y']}, AMM={crazy_df['y_rinc']}"
    
    assert np.abs(test_final['re_out_x'] - crazy_df['x_rout'].item()) < tolerance, \
        f"re_out_x mismatch: Test={test_final['re_out_x']}, AMM={crazy_df['x_rout']}"
    
    assert np.abs(test_final['re_out_y'] - crazy_df['y_rout'].item()) < tolerance, \
        f"re_out_y mismatch: Test={test_final['re_out_y']}, AMM={crazy_df['y_rout'].item()}"
    
    assert np.abs(test_final['dis_in_fee'] - crazy_df['dis_inc_fees'].item()) < tolerance, \
        f"dis_in_fee mismatch: Test={test_final['dis_in_fee']}, AMM={crazy_df['dis_inc_fees'].item()}, sigma={sigma}, gamma={gamma}"
    
    assert np.abs(test_final['dis_out_fee'] - crazy_df['dis_out_fees'].item()) < tolerance, \
        f"dis_out_fee mismatch: Test={test_final['dis_out_fee']}, AMM={crazy_df['dis_out_fees']}, sigma={sigma}, gamma={gamma}"
    
    # If we get here, all assertions passed
    print(f"All parameters match within tolerance for sigma={sigma}, gamma={gamma}!")

def test_simulator_results_match():
    """
    Test that Test_Simulator and AMMSimulator produce the same results
    given the same parameters.
    """
    # Test parameters
    params = {
        'sigma': 1.0,
        'gamma': 0.003,
        'dt': 1/(365*24),
        'steps': 100,
        'seed': 0,
        'tolerance': 1e-6
    }
    
    compare_simulators(params)

if __name__ == "__main__":
    # Test parameter combinations
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    gammas = [0.001, 0.002, 0.003, 0.004, 0.005]
    dt = 0.1
    steps = 1000
    seed = 0
    tolerance = 1e-6
    
    for sigma in sigmas:
        for gamma in gammas:
            params = {
                'sigma': sigma,
                'gamma': gamma,
                'dt': dt,
                'steps': steps,
                'seed': seed,
                'tolerance': tolerance
            }
            compare_simulators(params)
    
    # Run pytest if called directly
    import pytest
    pytest.main([__file__, "-v"])