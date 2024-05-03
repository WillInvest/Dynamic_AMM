# Generalized Automated Market Maker (AMM) Simulation

Welcome to the Generalized Automated Market Maker (AMM) Simulation repository! This Python project is designed to simulate the behavior of a generalized automated market maker, a crucial component of decentralized finance (DeFi) platforms. This simulation encompasses various modules such as AMM core functionality, fee structure implementation, solver algorithms, and utility functions to provide a comprehensive understanding of how AMMs operate.

## Prerequisites

- Python 3.10 or higher

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/hanlonlab/AMM-Python
   ```

2. Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

<!-- ## Project Structure

- `amm.py`: Contains the core logic for the Automated Market Maker.
- `fee_structure.py`: Implements the fee structure for trades within the AMM.
- `solver.py`: Provides algorithms for solving equations and optimizing AMM parameters.
- `utility_functions.py`: Includes utility functions used throughout the project. -->

## Usage

Updates on usage 5/3/34 - drew carranti

1. amm folder
- contains all amm infrastructure
- not much different from inmplementation as far as organization of logic into files and functions
2. Analysis
analysis folder
- contains Yuvaraj’s plotting code - kind of deprecated as of now bcs any plots are being generated in sim.ipynb for experiments
3. api_key
- for storing individual kaiko api key - no longer in use
data folder
- contains crypto_data folder and kaiko script for pulling data
experiment folder
- prev folder has old experimentation work - including previous files from Sean, Akashi, yuvaraj, that I have since integrated the accurate components into the greater codebase
- main - original main file we had previously been running out of
- sim script - when the notebook has been used to find the set of experiments we want to run and we want to run a lot of them at once - we can parameterize sim, move updated version into the script, and then called on repeatedly for different params
- sim.ipynb - used for developing experimentation framework before being parameterized and compressed into script for repeated calling
gbm
- contains gbm generation code - most is deprecated as it was primarily built out for kaiko gbm calibration - but we 




Check ```main.py```
<!-- To use the AMM simulation, import the necessary modules in your Python script and create instances of the AMM, fee structure, solver, and utility functions as needed. You can then simulate various scenarios and analyze the behavior of the AMM under different conditions.

```python
from amm import AMM
from fee_structure import FeeStructure
from solver import Solver
from utility_functions import UtilityFunctions

# Create instances of AMM, FeeStructure, Solver, and UtilityFunctions
amm_instance = AMM()
fee_structure_instance = FeeStructure()
solver_instance = Solver()
utility_functions_instance = UtilityFunctions()

# Simulate AMM behavior and analyze results
# ... (add your simulation code here)

# Example: Simulate a trade and calculate resulting balances
trade_amount = 100  # Amount of tokens to be traded
input_token_balance = 1000  # Initial balance of input token in the AMM
output_token_balance = 500  # Initial balance of output token in the AMM
trade_fee = 0.01  # Trade fee percentage

# Perform a trade using the AMM
output_amount = amm_instance.trade(input_token_balance, output_token_balance, trade_amount, trade_fee)

# Print the resulting output amount after the trade
print(f"Output amount after trade: {output_amount}")
``` -->

## Contributing

We welcome contributions from the community! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.

Happy simulating! 🚀