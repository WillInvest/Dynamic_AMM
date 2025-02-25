"""Runner script for two-step AMM analysis"""
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from two_step_exp.two_step_analysis import TwoStepAnalysis

if __name__ == "__main__":
    # Parameter setup
    sigmas = np.round(np.arange(0.1, 1, 0.1), 1)
    fee_rates = np.linspace(0.0001, 0.8, 10000)
    
    # Run analysis
    analyzer = TwoStepAnalysis()
    analyzer.calculate_metrics_parallel(
        sigmas=sigmas,
        fee_rates=fee_rates,
        output_path='output/two_step_metrics.parquet',
        n_workers=24
    ) 