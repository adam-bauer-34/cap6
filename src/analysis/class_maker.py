"""Class instance maker function.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
8.17.2022

This code contains a function which makes a list of relevant CAP6 model
classes for a list of run numbers. This is commonly used in Jupyter notebooks
which generate figures, and helps make those slimmer and easier to read.

NOTE: Notebook must be run in main directory, i.e., the directory *above*
/src!
"""

import numpy as np

from src.tree import TreeModel
from src.climate import BPWClimate
from src.emit_baseline import BPWEmissionBaseline
from src.damage import BPWDamage
from src.tools import import_csv

def make_class_instances(run_list):
    """Make class instance list.

    This function makes a list of classes for each run in run_list.

    Parameters
    ----------
    run_list: list
        list of run numbers, where run number refers to the number of the run
        in research_runs.csv

    Returns
    -------
    tree_list: list
        list of `TreeModel` objects for every run
    damage_list: list
        list of `BPWDamage` objects for every run
    climate_list: list
        list of `BPWClimate` objects for every run
    emit_baseline_list: list
        list of `BPWEmissionsBaseline` objects for every run
    """

    # import header, indicies, and data from reference .csv file
    data_csv_file = 'research_runs'
    header, indices, data = import_csv(data_csv_file, delimiter=',', indices=2)

    # instance classes
    tree_list = []
    damage_list = []
    climate_list = []
    emit_baseline_list = []

    for i in run_list:

        # import model params
        ra, eis, pref, growth, tech_chg, tech_scale, dam_func,\
            baseline_num, tip_on, bs_premium, d_unc, t_unc = data[i]

        baseline_num = int(baseline_num)
        dam_func = int(dam_func)
        tip_on = int(tip_on)
        d_unc = int(d_unc)
        t_unc = int(t_unc)

        # temporary tree model
        tmp_t = TreeModel(decision_times=[0, 10, 40, 80, 130, 180, 230],
                          prob_scale=1.0)

        # temporary emissions model
        tmp_be = BPWEmissionBaseline(tree=tmp_t, baseline_num=baseline_num)
        tmp_be.baseline_emission_setup()

        # temporary climate class
        draws = 3 * 10**6
        tmp_cl = BPWClimate(tmp_t, tmp_be, draws=draws)

        # temporary damage class
        d_m = 0.01
        mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]
        tmp_d = BPWDamage(tree=tmp_t, emit_baseline=tmp_be, climate=tmp_cl,
                          mitigation_constants=mitigation_constants,
                          draws=draws)

        # dam sim file
        damsim_filename = ''.join(["simulated_damages_df", str(dam_func),
                                   "_TP", str(tip_on), "_SSP",
                                   str(baseline_num), "_dunc", str(d_unc),
                                   "_tunc", str(t_unc)])

        tmp_d.import_damages(file_name=damsim_filename)

        # append to lists
        tree_list.append(tmp_t)
        damage_list.append(tmp_d)
        climate_list.append(tmp_cl)
        emit_baseline_list.append(tmp_be)

    return tree_list, damage_list, climate_list, emit_baseline_list
