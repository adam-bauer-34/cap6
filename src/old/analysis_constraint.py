"""Constraint analysis class.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
4/22/2022

This code contains the ConstraintAnalysis class, one of the family of analysis classes used in TCREZClimate.
"""

import numpy as np

from scipy.optimize import brentq, fmin
from .tools import import_csv, write_columns_csv
from .optimization import GradientSearch, GeneticAlgorithm

class ConstraintAnalysis(object):
    """Constraint analysis.

    Attributes
    ----------
    run_name: string
        the name of the run
    utility: `EZUtility` object
        utility class from TCREZClimate
    const_value: float
        ?
    opt_m:
        ?
    """
    def __init__(self, run_name, utility, const_value, opt_m=None):
        self.run_name = run_name
        self.utility = utility 
        self.cfp_m = self.constraint_first_period(utility, const_value, utility.tree.num_decision_nodes)
        self.opt_m = opt_m
        if self.opt_m is None:
            self.opt_m = self._get_optimal_m()

        self.con_cost = self._constraint_cost()
        self.delta_u = self._first_period_delta_udiff()

        self.delta_c = self._delta_consumption()
        self.delta_c_billions = self.delta_c * self.utility.cost.cons_per_ton \
                                * self.utility.damage.emit_baseline.emit_level[0]
        self.delta_emission_gton = self.opt_m[0]*self.utility.damage.emit_baseline.emit_level[0]
        self.deadweight = self.delta_c*self.utility.cost.cons_per_ton / self.opt_m[0]

        self.delta_u2 = self._first_period_delta_udiff2()
        self.marginal_benefit = (self.delta_u2 / self.delta_u) * self.utility.cost.cons_per_ton
        self.marginal_cost = self.utility.cost.price(0, self.cfp_m[0], 0)

    def _get_optimal_m(self):
        try:
            header, index, data = import_csv(self.run_name+"_node_period_output")
        except:
            print("No such file for the optimal mitigation..")
        return data[:, 0] 

    def _constraint_cost(self):
        opt_u = self.utility.utility(self.opt_m)
        cfp_u = self.utility.utility(self.cfp_m)
        return opt_u - cfp_u

    def _delta_consumption(self):
        return self.find_bec(self.cfp_m, self.utility, self.con_cost)

    def _first_period_delta_udiff(self):
        u_given_delta_con = self.utility.adjusted_utility(self.cfp_m, first_period_consadj=0.01)
        cfp_u = self.utility.utility(self.cfp_m)
        return u_given_delta_con - cfp_u

    def _first_period_delta_udiff2(self):
        m = self.cfp_m.copy()
        m[0] += 0.01
        u = self.utility.utility(m)
        cfp_u = self.utility.utility(self.cfp_m)
        return u - cfp_u
        
    def save_output(self, prefix=None):
        if prefix is not None:
            prefix += "_" 
        else:
            prefix = ""

        write_columns_csv([self.con_cost, [self.delta_c], [self.delta_c_billions], [self.delta_emission_gton],
                           [self.deadweight], self.delta_u, self.marginal_benefit, [self.marginal_cost]], 
                           prefix + self.run_name + "_constraint_output",
                          ["Constraint Cost", "Delta Consumption", "Delta Consumption $b", 
                           "Delta Emission Gton", "Deadweight Cost", "Marginal Impact Utility",
                           "Marginal Benefit Emissions Reduction", "Marginal Cost Emission Reduction"])

    def find_bec(m, utility, constraint_cost, a=-150, b=150):
        """Used to find a value for consumption that equalizes utility at time 0 in two different solutions.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        utility : `Utility` object
            object of utility class
        constraint_cost : float
            utility cost of constraining period 0 to zero
        a : float, optional
            initial guess
        b : float, optional
            initial guess - f(b) needs to give different sign than f(a)

        Returns
        -------
        tuple
            result of optimization

        Note
        ----
        requires the 'scipy' package
        """

        def min_func(delta_con):
            base_utility = utility.utility(m)
            new_utility = utility.adjusted_utility(m, first_period_consadj=delta_con)
            print(base_utility, new_utility, constraint_cost)
            return new_utility - base_utility - constraint_cost

        return brentq(min_func, a, b)

    def constraint_first_period(utility, first_node, m_size):
        """Calculate the changes in consumption, the mitigation cost component of consumption,
        and new mitigation values when constraining the first period mitigation to `first_node`.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        utility : `Utility` object
            object of utility class
        first_node : float
            value to constrain first period to
        
        Returns
        -------
        tuple
            (new mitigation array, storage tree of changes in consumption, ndarray of costs in first sub periods)

        """
        fixed_values = np.array([first_node])
        fixed_indicies = np.array([0])
        ga_model = GeneticAlgorithm(pop_amount=400, num_generations=200, cx_prob=0.8, mut_prob=0.5, bound=1.5,
                    num_feature=m_size, utility=utility, fixed_values=fixed_values,
                    fixed_indices=fixed_indicies, print_progress=True)

        gs_model = GradientSearch(var_nums=m_size, utility=utility, accuracy=1e-7,
                                iterations=200, fixed_values=fixed_values, fixed_indices=fixed_indicies,
                                print_progress=True)

        final_pop, fitness = ga_model.run()
        sort_pop = final_pop[np.argsort(fitness)][::-1]
        new_m, new_utility = gs_model.run(initial_point_list=sort_pop, topk=1)

        print("SCC and Utility after constrained gs: {}, {}".format(new_m[0], new_utility))  

        """
        u_f_calls=0
        
        def new_iu(m):
            global u_f_calls
            uu = -1.*utility.utility(m,return_trees=False)
            u_f_calls += 1
            if u_f_calls%500 == 0:
                print(u_f_calls, uu[0], m)
            return uu
        """
        u_f_calls = [0]

        def new_iu(m):
            uu = -1.*utility.utility(m, return_trees=False)
            u_f_calls[0] += 1
            if u_f_calls[0]%500 == 0:
                print(u_f_calls[0], uu[0], m)
            return uu

        newfmin_out = fmin(new_iu, new_m, xtol=5.e-5,maxfun=10**5,maxiter=2*(10**5),full_output=True)
        
        new_m = newfmin_out[0]
        new_utility = -1.0*newfmin_out[1]

        return new_m