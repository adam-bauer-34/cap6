"""Cost calibration parameter dictionary.

Adam Michael Bauer
adammb4@illinois.edu
University of Illinois at Urbana Champaign
1.18.2022
"""

# NOTE: First array is \tau parameters, second is \xi.

import numpy as np

cost_cal_params = {'no-free-lunches': np.array([[58.86058226462219, 58.8605820424876,
                                                58.860582004738724, 58.86058248081809,
                                                58.860581972135996], 
                                                [1.831952215967607, 2.2930010323253702,
                                                2.7872590206143646, 2.382232052452577,
                                                2.916435234033552]]),
                  'main-specification': np.array([[27.50355316561843, 27.503553614159248,
                                                27.503553963318577, 27.503553813217213,
                                                27.503556976032684], 
                                                [1.8864102504517255, 2.3611645411309974,
                                                2.8701152048906655, 2.4530481113692773,
                                                3.003131263293185]]),
                  }