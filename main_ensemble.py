"""CAP6 hypercube sampling run.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
8.19.2022

The idea here is to choose an emissions baseline, and then using a
Latin Hypercube sample the remaining parameters (RA, EIS, PRTP, tech_chg,
tech_scale). We always assume:
    tip_on = 1
    d_unc = 1
    t_unc = 1
    bs_premium = 10000
    dam_func = 0
    g = 0.015
    baseline_num = taken as command line input, but it's constant and either 1,
    2, 3, 4 or 5
    no_free_lunch = False

The rest have user defined ranges; default in this file are:
    ra = {3, 15}
    eis = {0.55, 1.08}
    pref = {0.001, 0.0147}
    tech_chg = {0, 3}
    tech_scale = {0, 3}

! See lines 88 and 89 to change these ranges.

To run: python BPW_lhc_sampling.py [baseline_num]
"""

import sys
import pathlib
import pprint
import os

import numpy as np

from src.tree import TreeModel
from src.emit_baseline import BPWEmissionBaseline
from src.cost import BPWCost
from src.climate import BPWClimate
from src.damage import BPWDamage
from src.utility import EZUtility
from src.optimization import  GeneticAlgorithm
from src.gen_samples import generate_samples

# extract baseline number from command line input
baseline_num = int(sys.argv[1])

"""Set parameters controlling the run type.
optimize being set to False usually implies that the damage simulation is being
tested.
"""

optimize = True
test_run = False
gen_samples = True
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
    # increase if you want more precision, but this generally works
    N_generations_ga = 100 

# set number of samples and dimension of sample space
N_RUNS = 1000
DIMS = 5

# get path to current file and set sample file name
path = str(pathlib.Path(__file__).parent.resolve())
samp_fname = ''.join([path, '/data/LHC_samples_N', str(N_RUNS), '_DIMS',
                      str(DIMS), '.csv'])

# generate samples (if necessary)
if gen_samples:
    # params here are: RA, EIS, tech_chg, tech_scale, prtp
    ubs = [15., 1.08, 3., 3., 0.0147]
    lbs = [3., 0.55, 0., 0., 0.001]
    generate_samples(N_RUNS, DIMS, lbs, ubs, save_file=True, filename=samp_fname)

# import parameter sample values
param_vals = np.genfromtxt(samp_fname, delimiter=',')
print("Parameter values imported successfully!")

# parse samples (make sure you check to make sure this is consistent with the
# ranges defined above!)
ras, eiss, exs, ends, prtps = param_vals.T
param_names = ["RA", "EIS", "tech_chg", "tech_scale", "PRTP"]

# make final list of expected prices and a run tracker (i)
exp_prices_and_uopts = np.zeros((N_RUNS, 6+1))
exp_mits = np.zeros((N_RUNS, 6))
exp_Ts = np.zeros_like(exp_mits, dtype=np.float32)
exp_concs = np.zeros_like(exp_mits, dtype=np.float32)
exp_dams = np.zeros_like(exp_mits, dtype=np.float32)
i = 0

"""Run TCREZClimate for runs for each parameter sampling above.

The below is basically the same as BPW_main.py, up until we save
the output.
"""

for (ra, eis, tech_chg, tech_scale, pref) in zip(ras, eiss, exs, ends, prtps):
    """Set constant model params.
    """
    dam_func = 0
    tip_on = 1
    d_unc = 1
    t_unc = 1
    growth = 0.015
    no_free_lunch = False

    print("Run number: ", i)
    pprint.pprint(set(zip(param_names, [ra, eis, tech_chg, tech_scale, pref])))

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
                backstop_premium=10000., no_free_lunch=no_free_lunch)

    """Damage class. We set draws to the number of Monte Carlo samples to take
    from damage distributions, such as TCRE. We also pass a list of constant
    values of mitigation for the damage simulation.
    """

    d_m = 0.01
    mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]
    df = BPWDamage(tree=t, emit_baseline=baseline_emission_model,
                   climate=climate, mitigation_constants=mitigation_constants,
                   draws=draws)

    """Run damages simulation, or import damages depending on the
    import_damages flag above.

    NOTE: import_damages has the optional argument for a filename. This is
    important if you're varying the damage function across simulations.

    The default is currently "BPW_simulated_damages_TCRE.csv". So if you don't
    specify damsim_filename, the default will be used. 
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
                                    utility=u, print_progress=False)

        """Run the genetic algorithm to find the final "population" of
        individuals and their fitness.
        """

        final_pop, fitness = ga_model.run()

        """Sort the population by their fitnes.
        """

        sort_pop = final_pop[np.argsort(fitness)][::-1]
        sort_fit = fitness[np.argsort(fitness)][::-1]
        m_opt = sort_pop[0]
        u_opt = sort_fit[0]

        # calculate the expected price and expected mitigation at each period,
        # as well as the optimal utility of the run to save for analysis later.
        price_node = np.zeros(len(m_opt))
        exp_m_node = np.zeros(len(m_opt))
        dam_node = np.zeros(len(m_opt))
        T_node = np.zeros(len(m_opt))
        conc_node = np.zeros(len(m_opt))

        exp_price_and_uopt = np.zeros(t.num_periods+1)
        exp_mit = np.zeros(t.num_periods)
        exp_T = np.zeros(t.num_periods)
        exp_conc = np.zeros(t.num_periods)
        exp_dam = np.zeros(t.num_periods)

        # put optimal utility at end for comparison later
        exp_price_and_uopt[-1] = u_opt

        # go through each period
        for period in range(t.num_periods):
            # number of years in period, number of years passing in each
            # period, nodes in the period, number of years passing in each
            # period again but list form
            years = t.decision_times[period]
            period_yrs = t.decision_times[period+1] - t.decision_times[period]
            nodes = t.get_nodes_in_period(period)
            period_lens = t.decision_times[:period+1]

            # now go through nodes and calculate the price at each, and then
            # the expected price at each period
            for node in range(nodes[0], nodes[1]+1):
                ave_mitigation = df.average_mitigation_node(m_opt, node, period)
                price_node[node] = c.price(years, m_opt[node], ave_mitigation)
                exp_m_node[node] = ave_mitigation
                dam_node[node] = df._damage_function_node(m_opt, node)
                conc_node[node] = climate.get_conc_at_node(m_opt, node)
                mit_emit, _  = baseline_emission_model.get_mitigated_baseline(m_opt,
                                                                          node=node,
                                                                          baseline='cumemit')
                T_node[node] = climate.TCRE_BEST_ESTIMATE * mit_emit[-1]

            # take expectations of a given parameter in each period
            probs = t.get_probs_in_period(period)
            exp_price_and_uopt[period] = np.dot(price_node[nodes[0]:nodes[1]+1], probs)
            exp_mit[period] = np.dot(exp_m_node[nodes[0]:nodes[1]+1], probs)
            exp_dam[period] = np.dot(dam_node[nodes[0]:nodes[1]+1], probs)
            exp_conc[period] = np.dot(conc_node[nodes[0]:nodes[1]+1], probs)
            exp_T[period] = np.dot(T_node[nodes[0]:nodes[1]+1], probs)

    # store in pre-defined arrays
    exp_prices_and_uopts[i] = exp_price_and_uopt
    exp_mits[i] = exp_mit
    exp_Ts[i] = exp_T
    exp_concs[i] = exp_conc
    exp_dams[i] = exp_dam
    i += 1

# save output! 
cwd = os.getcwd() # find current working directory
np.savetxt(''.join([cwd, "/data/lhc_sampling_paths_ssp", str(baseline_num), "_N",
                    str(N_RUNS), ".csv"]),
           exp_prices_and_uopts, delimiter=',')

np.savetxt(''.join([cwd, "/data/lhc_sampling_exp_m_ssp", str(baseline_num), "_N",
                    str(N_RUNS), ".csv"]),
           exp_mits, delimiter=',')

np.savetxt(''.join([cwd, "/data/lhc_sampling_exp_T_ssp", str(baseline_num), "_N",
                    str(N_RUNS), ".csv"]),
           exp_Ts, delimiter=',')

np.savetxt(''.join([cwd, "/data/lhc_sampling_exp_conc_ssp", str(baseline_num), "_N",
                    str(N_RUNS), ".csv"]),
           exp_concs, delimiter=',')

np.savetxt(''.join([cwd, "/data/lhc_sampling_exp_dam_ssp", str(baseline_num), "_N",
                    str(N_RUNS), ".csv"]),
           exp_dams, delimiter=',')
