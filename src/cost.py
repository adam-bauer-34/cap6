"""Cost classes.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.14.2022

This code contains two classes. The first is an abstract Cost class to be used
as a template for other cost classes. The second is our cost class -- BPWCost
-- which computes the cost of carbon.
"""

import numpy as np
from abc import ABCMeta, abstractmethod

from .cal.cost_cal import cost_cal_params

class Cost(object, metaclass=ABCMeta):
    """Abstract Cost class for the EZ-Climate model.
    """

    @abstractmethod
    def cost(self):
        pass

    @abstractmethod
    def price(self):
        pass

class BPWCost(Cost):
    """Class to evaluate the cost curve for the EZ-Climate model.

    Parameters
    ----------
    tree : `TreeModel` object
        tree structure used
    emit_at_0 : float
        initial GHG emission level
    baseline_num: int
        tells what emissions baseline we're using
    tech_const: float
        rate of enxogeneous technological development
    tech_scale: float
        rate of exdogeneous technological development
    cons_at_0 : float
        initial consumption. Default $61880 bn based on US 2020 values.
    backstop_premium: float
        premium tax on co2 removal
    no_free_lunch: bool
        "No free lunch" calibration on?

    Attributes
    ----------
    tree: `TreeModel` object
        tree structure used
    taus: list
        list of tau_0 values that were fit to IPCC AR6 WGIII data, see the
        paper for details
    powers: list
        list of power values (greek letter xi in paper) values that were fit to
        IPCC AR6 WGIII data, see the paper for details
    tau_0: float
        price of emitting all emissions from baseline
    power: float
        power law scaling of cost curve
    tech_const: float
        rate of exogeneous technological development
    tech_scale: float
        rate of endogeneous technological development
    cons_per_ton : float
        consumption per tonne of CO2 emitted
    backstop_premium: float
        premium tax on co2 removal
    no_free_lunch: bool
        "No free lunch" calibration on?
    """

    def __init__(self, tree, emit_at_0, baseline_num, tech_const, tech_scale,
                 cons_at_0, backstop_premium, no_free_lunch):
        self.tree = tree
        if no_free_lunch:
            self.taus, self.powers = cost_cal_params['no-free-lunches']
        else:
            self.taus, self.powers = cost_cal_params['main-specification']

        self.tau_0 = self.taus[baseline_num - 1]
        self.power = self.powers[baseline_num - 1]
        self.tech_const = tech_const
        self.tech_scale = tech_scale
        self.cons_per_ton = cons_at_0 / emit_at_0
        self.backstop_premium = backstop_premium

    def cost(self, period, mitigation, ave_mitigation):
        """Calculates the mitigation cost for the period. For details about the
        cost function see our paper.

        Parameters
        ----------
        period : int
            period in tree for which mitigation cost is calculated
        mitigation : ndarray
            current mitigation values for period
        ave_mitigation : ndarray
            average mitigation up to this period for all nodes in the period

        Returns
        -------
        ndarray :
            cost
        """

        if mitigation.min() < 0.0:
            m0 = np.maximum(mitigation,0.)
        else:
            m0 = mitigation

        years = self.tree.decision_times[period]
        tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation)
                            / 100.0))**(years-10)
        mitigation_cost = self.tau_0 * ((np.exp(self.power * m0) - 1)\
                          * self.power**(-1) -  m0)

        # charge a premium for over-mitigating
        over_mit = m0 > 1.0
        if over_mit.any():
            mitigation_cost[over_mit] = (self.tau_0 + self.backstop_premium)\
                                      * ((np.exp(self.power * m0[over_mit])-1)\
                                      * self.power**(-1) - m0[over_mit])

        c = (mitigation_cost * tech_term) / self.cons_per_ton
        return c

    def price(self, years, mitigation, ave_mitigation):
        """Inverse of the cost function. Gives emissions price for any given
        degree of mitigation, average_mitigation, and horizon.

        Parameters
        ----------
        years : int y
            years of technological change so far
        mitigation : float
            mitigation value in node
        ave_mitigation : float
            average mitigation up to this period

        Returns
        -------
        float
            the price.
        """

        tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation)\
                            / 100))**(years-10)
        price = tech_term * self.tau_0 * (np.exp(mitigation*self.power) - 1)

        # charge premium for over mitigating
        over_mit = mitigation > 1.0
        if over_mit.any():
            price = tech_term * (self.tau_0 + self.backstop_premium)\
                  * (np.exp(mitigation[over_mit]*self.power) - 1)

        return price
