"""Damage calibration parameter dictionary.

Adam Michael Bauer
adammb4@illinois.edu
University of Illinois at Urbana Champaign
10.21.2022
"""

import numpy as np

dam_cal_params = {'burke_ssp1': np.array([0.075, 0.01, 2.5, 0.2, 0.04, 0.87]),
                  'burke_ssp2': np.array([0.065, 0.01, 2.0, 0.195, 0.04, 0.75]),
                  'burke_ssp3': np.array([0.062, 0.01, 2.0, 0.19, 0.04, 0.69]),
                  'burke_ssp4': np.array([0.049, 0.01, 2.5, 0.13, 0.04, 0.82]),
                  'burke_ssp5': np.array([0.065, 0.01, 2.1, 0.155, 0.04, 0.82]),
                  'structural': np.array([0.027, 0.01, 2.8]),
                  'meta-analytic': np.array([0.063, 0.022, 3.3])
                  }
