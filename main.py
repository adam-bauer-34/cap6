"""CAP6 main file.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.14.2022

This is the main file for the Climate Asset Pricing model -- AR6. 
Generates as many output files as runs desired.

To run: python BPW_main.py
"""

import pickle
import pprint
import os

import numpy as np

from src.tree import TreeModel
from src.emit_baseline import BPWEmissionBaseline
from src.cost import BPWCost
from src.climate import BPWClimate
from src.damage import BPWDamage
from src.utility import EZUtility
from src.analysis.climate_output import ClimateOutput
from src.tools import import_csv
from src.optimization import  GeneticAlgorithm, GradientSearch

"""Set parameters controlling the run type.
optimize being set to False usually implies that the damage simulation is being
tested.
"""

optimize = True
test_run = False
import_damages = False

"""If test_run, set optimization parameters to small values to run more
efficiently. Else, it's a full run, so crank those numbers up!
"""

if test_run:
    print("\n\n***WARNING --- RUNNING WITH LIMITED NUMBER OF ITERATIONS FOR\
          TEST PURPOSES***\n\n")
    N_generations_ga = 2
    N_iters_gs = 2
else:
    N_generations_ga = 150
    N_iters_gs = 100

"""Import header, indices, and data from reference .csv file.
"""

data_csv_file = 'research_runs'
header, indices, data = import_csv(data_csv_file, delimiter=',', indices=2)

"""Define runs of interest. To change which runs the code does, change
desired_runs. These numbers correspond to the run numbers in data_csv_file.
"""

desired_runs = [0]
for i in desired_runs:
    name = indices[i][1]

    """Parse model parameters from data.
    """

    ra, eis, pref, growth, tech_chg, tech_scale, dam_func,\
        baseline_num, tip_on, bs_premium, d_unc, t_unc,\
        no_free_lunch = data[i]

    baseline_num = int(baseline_num)
    dam_func = int(dam_func)
    tip_on = int(tip_on)
    d_unc = int(d_unc)
    t_unc = int(t_unc)
    no_free_lunch = int(no_free_lunch)

    print('**Running job:       ', name, '\n**Model Parameters are:')
    model_params = [ra, eis, pref, growth, tech_chg, tech_scale,\
                    dam_func, baseline_num, tip_on, bs_premium, d_unc,
                    t_unc, no_free_lunch]
    pprint.pprint(set(zip(header,model_params)))

    """Initialize model classes. First is the tree model.
    """

    t = TreeModel(decision_times=[0, 10, 40, 80, 130, 180, 230],
                  prob_scale=1.0)

    """Emission baseline model. We also run its setup function.
    """

    baseline_emission_model = BPWEmissionBaseline(tree=t,
                                                  baseline_num=baseline_num)
    baseline_emission_model.baseline_emission_setup()

    """Climate class. We set draws to the number of Monte Carlo samples to take
    from damage distributions, such as TCRE, and if the floor is on.
    """

    draws = 3 * 10**6
    climate = BPWClimate(t, baseline_emission_model, draws=draws)

    """Cost class to calculate the cost of carbon once damages are known.
    """

    c = BPWCost(t, emit_at_0=baseline_emission_model.baseline_gtco2[1],
                baseline_num=baseline_num, tech_const=tech_chg,
                tech_scale=tech_scale, cons_at_0=61880.0,
                backstop_premium=bs_premium, no_free_lunch=no_free_lunch)

    """Damage class. We set draws to the number of Monte Carlo samples to take
    from damage distributions, such as TCRE. We also pass a list of constant
    values of mitigation for the damage simulation.
    """

    d_m = 0.1
    mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]
    df = BPWDamage(tree=t, emit_baseline=baseline_emission_model,
                   climate=climate, mitigation_constants=mitigation_constants,
                   draws=draws)

    """Run damage simulation or import damages you've already made.

    NOTE: import_damages has the optional argument for a filename. This is
    important if you're varying the damage function across simulations.

    The default is currently "BPW_simulated_damages_TCRE.csv".
    """

    damsim_filename = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline_num),
                               "_dunc", str(d_unc), "_tunc", str(t_unc)])

    if import_damages:
        df.import_damages(file_name=damsim_filename)

    else:
        df.damage_simulation(filename=damsim_filename, save_simulation=True,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)

    """The economic utility class.
    """

    u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, eis=eis, ra=ra,
                  time_pref=pref, cons_growth=growth)

    """Now run the optimzation routine. This consists of a genetic algorithm
    and gradient search. The result of this procedure is the optimal mitigation
    vector and optimal economic utility. These are then used to calculate the
    cost of carbon.
    """

    if optimize:
        """Make instances of the genetic algorithm (ga) and the gradient search
        (gs) algorithms.
        """

        ga_model = GeneticAlgorithm(pop_amount=400,
                                    num_generations=N_generations_ga,
                                    cx_prob=0.8, mut_prob=0.50, bound=1.5,
                                    num_feature=t.num_decision_nodes,
                                    utility=u, print_progress=True)

        gs_model = GradientSearch(var_nums=63, utility=u, accuracy=5.e-7,
                        iterations=N_iters_gs, print_progress=True)

        """Run the genetic algorithm to find the final "population" of
        individuals and their fitness.
        """

        final_pop, fitness = ga_model.run()

        """Sort the population by their fitnes.
        """

        sort_pop = final_pop[np.argsort(fitness)][::-1]

        """Use gradient search to calculate the optimal mitigation vector and
        the optimal economic utility.
        """
        m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)
        print("SCC and Utility after gs:   ", c.price(0, m_opt[0], 0), u_opt)

        """Save the output of the gradient search. First make a dictionary
        which stores the damage recombined tree values, the optimal mitigation
        vector and the optimal economic utility.
        """

        p = {}
        p['df.d_rcomb'] = df.d_rcomb
        p['m_opt'] = m_opt
        p['u_opt'] = u_opt

        print("Saving parameters to pickle file:")
        picklename = 'data/'+name+'_log.pickle'
        with open(picklename, 'wb') as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('    parameter file saved as '+picklename)

        """Now analyze the results using the Risk Decomposition and Climate Output
        classes.
        """

        filelist = [f for f in os.listdir('data/') if f.startswith(name) and
                    f.endswith('.csv')]
        for f in filelist:
            print('Removing file','data/'+f)
            os.remove('data/'+f)

        print("Analyzing output and saving!")

        """Instantiate and use the climate output class.
        """

        co = ClimateOutput(u)
        co.calculate_output(m_opt)
        co.save_output(m_opt, prefix=name)
