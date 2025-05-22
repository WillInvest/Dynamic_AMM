"""Runner script for two-step AMM analysis"""
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from two_step_exp.two_step_analysis import TwoStepAnalysis

if __name__ == "__main__":
    import os
    path = '/home/shiftpub/Dynamic_AMM/two_step_exp/output'
    os.makedirs(path, exist_ok=True)
    
    # Parameter setup
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fee_rates = np.arange(0.0005, 0.9005, 0.01)
    # Run analysis
    analyzer = TwoStepAnalysis(ell_r=1, ell_s=1)
    analyzer.calculate_metrics_parallel(
        sigmas=sigmas,
        fee_rates=fee_rates,
        output_path=f'{path}/two_step_metrics.parquet',
        n_workers=24
    ) 