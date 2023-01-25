"""Generate samples from a Latin Hypercube for CAP6 'ensamble' model runs.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
8.19.2022

This code contains a function which makes a text file that contains model
parameter values that get sampled in the BPW_lhc_sampling.py file.
"""

import numpy as np

from scipy.stats import qmc

def generate_samples(N_RUNS, DIMS, lbs, ubs, save_file=True,
                     filename="/data/BPW_LHC_samples.csv"):
    """Generate samples of model parameters from a Latin Hypercube.

    This function generates a set of samples from a Latin Hypercube. The
    sample 0 (i.e., the 0th sample) ranges from lbs[0] to ubs[0], and so on.

    Parameters
    ----------
    N_RUNS: int
        number of total samples desired
    DIMS: int
        dimensionality of sample space (how many parameter ranges do you want to
        generate?)
    lbs: list
        lower bounds of each parameter range
    ubs: list
        upper bounds of each parameter range
    save: bool (default = True)
        save output to .csv file?
    filename: string (default is "BPW_LHC_samples.csv" in the /data
    subdirectory)

    Returns
    -------
    if save is False; returns:
        scaled_sample: (N_RUNS, DIMS) numpy array of parameter values
    """

    sampler = qmc.LatinHypercube(d=DIMS)
    sample = sampler.random(n=N_RUNS)
    scaled_sample = qmc.scale(sample, lbs, ubs)

    if save_file:
        np.savetxt(filename, scaled_sample, delimiter=',')
        print("Samples drawn and saved!")

    else:
        print("Samples drawn and returned!")
        return scaled_sample
