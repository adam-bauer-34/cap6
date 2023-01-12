"""Risk decomposition class.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
4/22/2022

This code contains the class RiskDecomposition, which is part of the family of analysis classes in TCREZClimate.
"""

import numpy as np

from scipy.optimize import brentq
from ..storage_tree import BigStorageTree
from ..tools import write_columns_csv, append_to_existing

class RiskDecomposition(object):
    """Calculate and save analysis of output from the EZ-Climate model.

    Parameters
    ----------
    utility : `Utility` object
        object of utility class

    Attributes
    ----------
    utility : `Utility` object
        object of utility class
    sdf_tree : `BaseStorageTree` object
        SDF for each node
    expected_damages : ndarray
        expected damages in each period
    risk_premiums : ndarray
        risk premium in each period
    expected_sdf : ndarray
        expected SDF in each period
    cross_sdf_damages : ndarray
        cross term between the SDF and damages
    discounted_expected_damages : ndarray
        expected discounted damages for each period
    net_discount_damages : ndarray
        net discount damage, i.e. when cost is also accounted for
    cov_term : ndarray 
        covariance between SDF and damages
    """
    
    def __init__(self, utility):
        self.utility = utility
        self.sdf_tree = BigStorageTree(utility.period_len, utility.tree.decision_times)
        self.sdf_tree.set_value(0, np.array([1.0]))

        n = len(self.sdf_tree)
        self.expected_damages = np.zeros(n)
        self.risk_premiums = np.zeros(n)
        self.expected_sdf = np.zeros(n)
        self.cross_sdf_damages = np.zeros(n)
        self.discounted_expected_damages = np.zeros(n)
        self.net_discount_damages = np.zeros(n)
        self.cov_term = np.zeros(n)
        self.expected_sdf[0] = 1.0

    def save_output(self, m, prefix=None):
        """Save attributes calculated in `sensitivity_analysis` into the file prefix + `sensitivity_output`
        in the `data` directory in the current working directory.

        Furthermore, the perpetuity yield, the discount factor for the last period is calculated, and SCC,
        expected damage and risk premium for the first period is calculated and saved in into the file
        prefix + `tree` in the `data` directory in the current working directory. If there is no `data` directory, 
        one is created.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        prefix : str, optional
            prefix to be added to file_name
        """

        end_price = self._find_term_structure(m, 0.01)
        perp_yield = self._perpetuity_yield(end_price, self.sdf_tree.periods[-2])

        damage_scale = self.utility.cost.price(0, m[0], 0) / (self.net_discount_damages.sum()+self.risk_premiums.sum())
        scaled_discounted_ed = self.net_discount_damages * damage_scale
        scaled_risk_premiums = self.risk_premiums * damage_scale

        if prefix is not None:
            prefix += "_" 
        else:
            prefix = ""

        write_columns_csv([self.expected_sdf, self.net_discount_damages, self.expected_damages, self.risk_premiums, 
                       self.cross_sdf_damages, self.discounted_expected_damages, self.cov_term, 
                       scaled_discounted_ed, scaled_risk_premiums], prefix + "sensitivity_output",
                           ["Year", "Discount Prices", "Net Expected Damages", "Expected Damages", "Risk Premium",
                            "Cross SDF & Damages", "Discounted Expected Damages", "Cov Term", "Scaled Net Expected Damages",
                            "Scaled Risk Premiums"], [self.sdf_tree.periods.astype(int)+2020]) 

        append_to_existing([[end_price], [perp_yield], [scaled_discounted_ed.sum()], [scaled_risk_premiums.sum()], 
                    [self.utility.cost.price(0, m[0], 0)]], prefix+"sensitivity_output",
                    header=["Zero Bound Price", "Perp Yield", "Expected Damages", "Risk Premium", 
                            "SCC"], start_char='\n')
        
        self._store_trees(prefix=prefix, tree_dict={'SDF':self.sdf_tree})

    def _find_term_structure(self, m, payment, a=0.0, b=1.5): 
        """Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of consumption in the final period. The purpose of this function is to find the interest rate embedded in the `EZUtility` model. 

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        payment : float
            value added to consumption in the final period
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

        def min_func(price):
            period_cons_eps = np.zeros(int(self.utility.tree.decision_times[-1]/self.utility.period_len) + 1)
            period_cons_eps[-2] = payment
            utility_with_payment = self.utility.adjusted_utility(m, period_cons_eps=period_cons_eps)
            first_period_eps = payment * price
            utility_with_initial_payment = self.utility.adjusted_utility(m, first_period_consadj=first_period_eps)
            return  utility_with_payment - utility_with_initial_payment

        return brentq(min_func, a, b)

    def _perpetuity_yield(self, price, start_date, a=0.1, b=100000):
        """Find the yield of a perpetuity starting at year `start_date`.

        Parameters
        ----------
        price : float
            price of bond ending at `start_date`
        start_date : int
            start year of perpetuity
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
        
        def min_func(perp_yield):
            return price - (100. / (perp_yield+100.))**start_date * (perp_yield + 100)/perp_yield

        return brentq(min_func, a, b)

    def _store_trees(self, prefix=None, start_year=2020, tree_dict={}):
        """Saves values of `BaseStorageTree` objects.

        The file is saved into the 'data' directory in the current working
        directory. If there is no 'data' directory, one is created.

        Parameters
        ----------
        prefix : str, optional
            prefix to be added to file_name
        start_year : int, optional
            start year of analysis
        tree_dict: dictionary with arbitrary keys and `BigStorageTree` entries
            dictionary of trees, where the key identifies which tree and content is a `BigStorageTree` object.
        """

        if prefix is None:
            prefix = ""
        for name in tree_dict.keys():
            tree_dict[name].write_columns(prefix + "trees", name, start_year)
