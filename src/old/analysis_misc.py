"""Miscellaneous analysis functions.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
4/22/2022

This code contains miscellaneous analysis functions that were never called in the original EZClimate. It is unclear at this time if they are depricated or just not useful.
"""

from scipy.optimize import brentq

def find_ir(m, utility, payment, a=0.0, b=1.0): 
    """Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of 
    consumption in the final period. The purpose of this function is to find the interest rate 
    embedded in the `EZUtility` model. 

    Parameters
    ----------
    m : ndarray or list
        array of mitigation
    utility : `Utility` object
        object of utility class
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
        utility_with_final_payment = utility.adjusted_utility(m, final_cons_eps=payment)
        first_period_eps = payment * price
        utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
        return utility_with_final_payment - utility_with_initial_payment

    return brentq(min_func, a, b)