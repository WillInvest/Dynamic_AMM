import os
import platform
import socket

def get_base_path():
    """
    Determine the base path based on the machine we're running on.
    Returns the appropriate base path for the current machine.
    """
    hostname = socket.gethostname()
    
    # Machine-specific paths
    paths = {
        # Add your local machine's hostname
        'haofu': '/Users/haofu/Desktop/AMM/Dynamic_AMM',
        # Add the server's hostname
        'shiftpub': '/home/shiftpub/Dynamic_AMM'
    }
    
    # Get the path from environment variable if set, otherwise use the machine-specific path
    base_path = os.getenv('DYNAMIC_AMM_PATH', paths.get(hostname))
    
    # If we can't determine the path, raise an error
    if not base_path:
        raise ValueError(f"Unknown hostname: {hostname}. Please either:\n"
                       f"1. Add your machine to the paths dictionary in config.py\n"
                       f"2. Set the DYNAMIC_AMM_PATH environment variable")
    
    return base_path

def get_output_dir(timestamp):
    """
    Get the output directory path for simulation results.
    """
    base_path = get_base_path()
    return os.path.join(base_path, 'inf_step_exp', 'mc_approach', 'crazy_simulation_results', timestamp)
