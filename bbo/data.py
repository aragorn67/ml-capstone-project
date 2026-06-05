"""
bbo/data.py — Data loading and management

Handles loading initial .npy files and appending weekly results.
"""

import numpy as np
import os


def load_function_data(func_num, new_data, base_path):
    """
    Load initial data for a function and append any new weekly results.

    WHAT: Combines the original .npy files (provided at start of challenge)
    with the results we've accumulated from weekly submissions.

    Args:
        func_num:  which function (1-8)
        new_data:  dict mapping function number to list of (query, output) pairs
        base_path: path to the initial_data folder

    Returns:
        inputs:  numpy array of shape (n_points, n_dims)
        outputs: numpy array of shape (n_points,)
    """
    folder = os.path.join(base_path, f"function_{func_num}")
    inputs = np.load(os.path.join(folder, "initial_inputs.npy"))
    outputs = np.load(os.path.join(folder, "initial_outputs.npy"))

    # Append weekly results if any exist
    if func_num in new_data and len(new_data[func_num]) > 0:
        new_inputs = np.array([pair[0] for pair in new_data[func_num]])
        new_outputs = np.array([pair[1] for pair in new_data[func_num]])
        inputs = np.vstack([inputs, new_inputs])
        outputs = np.concatenate([outputs, new_outputs])

    return inputs, outputs
