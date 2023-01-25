"""Utility class.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.21.2022

Utility class used for calculating the economic utility and related
quantities in CAP6. We implement Epstein-Zin preferences for
our utility calculations below.
"""

import numpy as np

from src.storage_tree import BigStorageTree, SmallStorageTree

np.seterr(all='ignore')

class EZUtility(object):
    """Calculation of Epstein-Zin utility for the CAP6 model.

    The Epstein-Zin utility allows for different rates of substitution across
    time and states. For specification see DLW-paper (2017) and BPW-paper (2023).

    Parameters
    ----------
    tree : `TreeModel` object
        tree structure used
    damage : `Damage` object
        class that provides damage methods
    cost : `Cost` object
        class that provides cost methods
    period_len : float
        subinterval length
    eis : float, optional
        elasticity of intertemporal substitution
    ra : float, optional
        risk-aversion
    time_pref : float, optional
        pure rate of time preference

    Attributes
    ----------
    tree : `TreeModel` object
        tree structure used
    damage : `Damage` object
        class that provides damage methods
    cost : `Cost` object
        class that provides cost methods
    period_len : float
        subinterval length
    cons_growth : float
        consumption growth
    growth_term : float
        1 + cons_growth
    r : float
        the parameter rho from the DLW-paper
    a : float
        the parameter alpha from the DLW-paper
    b : float
        the parameter beta from the DLW-paper
    potential_cons: float
        not sure, but seems like potential_cons = (1 + g)^t, where g is
        constant growth rate.
    """

    def __init__(self, tree, damage, cost, period_len, eis=0.9, ra=7.0,
                 time_pref=0.005, cons_growth=0.015):
        self.tree = tree
        self.damage = damage
        self.cost = cost
        self.period_len = period_len
        self.cons_growth = cons_growth
        self.growth_term = 1.0 + self.cons_growth
        self.r = 1.0 - 1.0/eis
        self.a = 1.0 - ra
        self.b = (1.0-time_pref)**period_len
        self.potential_cons = (np.ones(self.tree.decision_times.shape) \
                               + self.cons_growth)**self.tree.decision_times

    def utility(self, m, return_trees=False):
        """Calculating utility for the specific mitigation decisions `m`.

        Parameters
        ----------
        m : ndarray or list
            array of mitigations
        return_trees : bool
            True if method should return trees calculated in producing the utility

        Returns
        -------
        ndarray or tuple
            tuple of `BaseStorageTree` if return_trees else ndarray with utility at period 0

        Examples
        ---------
        Assuming we have declared a EZUtility object as 'ezu' and have a mitigation array 'm'

        >>> ezu.utility(m)
        array([ 9.83391921])
        >>> tree_dict = ezu.utility(m, return_trees=True)
        """

        utility_tree = BigStorageTree(subinterval_len=self.period_len,
                                      decision_times=self.tree.decision_times)
        cons_tree = BigStorageTree(subinterval_len=self.period_len,
                                   decision_times=self.tree.decision_times)
        ce_tree = BigStorageTree(subinterval_len=self.period_len,
                                 decision_times=self.tree.decision_times)
        cost_tree = SmallStorageTree(decision_times=self.tree.decision_times)

        self._end_period_utility(m, utility_tree, cons_tree, cost_tree)

        # makes generator object and iterates over it to fill the utility tree
        # with values in each period 
        it = self._utility_generator(m, utility_tree, cons_tree, cost_tree, ce_tree)
        for u, period in it:
            utility_tree.set_value(period, u)

        if return_trees:
            return {'Utility':utility_tree, 'Consumption':cons_tree,
                    'Cost':cost_tree, 'CertainEquivalence':ce_tree}
        # returns first value
        return utility_tree[0]

    def _end_period_utility(self, m, utility_tree, cons_tree, cost_tree):
        """Calculate the terminal utility.

        Calculates the utility in the final period and stores the values in the
        utility_tree object.

        Parameters
        ----------
        m: nd array
            Array of mitigation valus
        utility_tree: `BigStorageTree` object
            storage tree of utility values
        cons_tree: `BigStorageTree` object
            storage tree of consumption values
        cost_tree: `SmallStorageTree` object
            storage tree of cost values
        """

        # calc average mitigation and damages in the period 
        period_ave_mitigation = self.damage.average_mitigation(m,
                                                               self.tree.num_periods,
                                                               is_last=True)
        period_damage = self.damage.damage_function(m,
                                                    self.tree.num_periods,
                                                    is_last=True)

        # get a tuple of nodes in period
        damage_nodes = self.tree.get_nodes_in_period(self.tree.num_periods)

        # mitigation in the period
        period_mitigation = m[damage_nodes[0]:damage_nodes[1]+1]

        # calc cost in period, store value, and calculate the remaining values
        period_cost = self.cost.cost(self.tree.num_periods, period_mitigation,
                                     period_ave_mitigation)
        continuation = (1.0 / (1.0 - self.b*(self.growth_term**self.r)))**(1.0/self.r)

        cost_tree.set_value(cost_tree.last_period, period_cost)
        period_consumption = self.potential_cons[-1] * (1.0 - period_damage)
        period_consumption[period_consumption<=0.0] = 1e-18
        cons_tree.set_value(cons_tree.last_period, period_consumption)
        utility_tree.set_value(utility_tree.last_period,
                               (1.0 - self.b)**(1.0/self.r) * cons_tree.last \
                               * continuation)

    def _utility_generator(self, m, utility_tree, cons_tree, cost_tree,
                           ce_tree, cons_adj=0.0):
        """Generator fora calculating utility for each utility period besides
        the terminal utility.

        Parameters
        ----------
        m: nd array
            Array of mitigation valus
        utility_tree: `BigStorageTree` object
            storage tree of utility values
        cons_tree: `BigStorageTree` object
            storage tree of consumption values
        cost_tree: `SmallStorageTree` object
            storage tree of cost values
        ce_tree: `BigStorageTree` object
            storage tree of certain equivalence values
        cons_adj: float
            constant adjustment for first period utility
        """

        periods = utility_tree.periods[::-1]

        for period in periods[1:]:
            damage_period = utility_tree.between_decision_times(period)
            cert_equiv = self._certain_equivalence(period, damage_period, utility_tree)

            if utility_tree.is_decision_period(period+self.period_len):
                damage_nodes = self.tree.get_nodes_in_period(damage_period)
                period_mitigation = m[damage_nodes[0]:damage_nodes[1]+1]
                period_ave_mitigation = self.damage.average_mitigation(m, damage_period)
                period_cost = self.cost.cost(damage_period, period_mitigation,
                                             period_ave_mitigation)
                period_damage = self.damage.damage_function(m, damage_period)
                cost_tree.set_value(cost_tree.index_below(period+self.period_len),
                                    period_cost)

            period_consumption = self.potential_cons[damage_period] \
                                    * (1.0 - period_damage) * (1.0 - period_cost)
            period_consumption[period_consumption <= 0.0] = 1e-18

            if not utility_tree.is_decision_period(period):
                next_consumption = cons_tree.get_next_period_array(period)
                segment = period - utility_tree.decision_times[damage_period]
                interval = segment + utility_tree.subinterval_len

                if utility_tree.is_decision_period(period+self.period_len):
                    if period < utility_tree.decision_times[-2]:
                        next_cost = cost_tree[period+self.period_len]
                        next_consumption *= (1.0 - np.repeat(period_cost,2)) / (1.0 - next_cost)
                        next_consumption[next_consumption<=0.0] = 1e-18

                if period < utility_tree.decision_times[-2]:
                    temp_consumption = next_consumption/np.repeat(period_consumption,2)
                    period_consumption = np.sign(temp_consumption)*(np.abs(temp_consumption)**(segment/float(interval))) \
                                         * np.repeat(period_consumption,2)
                else:
                    temp_consumption = next_consumption/period_consumption
                    period_consumption = np.sign(temp_consumption)*(np.abs(temp_consumption)**(segment/float(interval))) \
                                         * period_consumption
            if period == 0:
                period_consumption += cons_adj

            ce_term = self.b * cert_equiv**self.r
            ce_tree.set_value(period, ce_term)
            cons_tree.set_value(period, period_consumption)
            u = ((1.0-self.b)*period_consumption**self.r + ce_term)**(1.0/self.r)
            yield u, period

    def _certain_equivalence(self, period, damage_period, utility_tree):
        """Calculate ceartainty equivalence utility.

        If we are between decision nodes, i.e. no branching, then certainty
        equivalent utility at time period depends only on the utility next
        period given information known today. Otherwise the certainty
        equivalent utility is the ability weighted sum of next period utility
        over the partition reachable from the state.

        Parameters
        ----------
        period: int
            The period we are at
        damage_period: nd array
            array of damages for each node in the period, sorted from worst to best
        utility_tree: `BigStorageTree` object
            tree which stores all utility values
        """

        if utility_tree.is_information_period(period):
            damage_nodes = self.tree.get_nodes_in_period(damage_period+1)
            probs = self.tree.node_prob[damage_nodes[0]:damage_nodes[1]+1]
            even_probs = probs[::2]
            odd_probs = probs[1::2]
            even_util = ((utility_tree.get_next_period_array(period)[::2])**self.a) * even_probs
            odd_util = ((utility_tree.get_next_period_array(period)[1::2])**self.a) * odd_probs
            ave_util = (even_util + odd_util) / (even_probs + odd_probs)
            cert_equiv = ave_util**(1.0/self.a)
        else:
            # no branching implies certainty equivalent utility at time period depends only on
            # the utility next period given information known today
            cert_equiv = utility_tree.get_next_period_array(period)

        return cert_equiv

    def adjusted_utility(self, m, period_cons_eps=None, node_cons_eps=None,
                         final_cons_eps=0.0, first_period_consadj=0.0,
                         return_trees=False):
        """Calculating aadjusted utility for sensitivity analysis.

        Used e.g. to find zero-coupon bond price.
        Values in parameters are used to adjust utility in different ways.

        Parameters
        ----------
        m : ndarray
            array of mitigations
        period_cons_eps : ndarray, optional
            array of increases in consumption per period
        node_cons_eps : `SmallStorageTree`, optional
            increases in consumption per node
        final_cons_eps : float, optional
            value to increase the final utilities by
        first_period_consadj : float, optional
            value to increase consumption at period 0 by
        return_trees : bool, optional
            True if method should return trees calculated in producing the
            utility

        Returns
        -------
        ndarray or tuple
            tuple of `BaseStorageTree` if return_trees else ndarray with utility at period 0

        Examples
        ---------
        Assuming we have declared a EZUtility object as 'ezu' and have a mitigation array 'm'

        >>> ezu.adjusted_utility(m, final_cons_eps=0.1)
        array([ 9.83424045])
        >>> tree_dict = ezu.adjusted_utility(m, final_cons_eps=0.1, return_trees=True)

        >>> arr = np.zeros(int(ezu.decision_times[-1]/ezu.period_len) + 1)
        >>> arr[-1] = 0.1
        >>> ezu.adjusted_utility(m, period_cons_eps=arr)
        array([ 9.83424045])

        >>> bst = BigStorageTree(5.0, [0, 15, 45, 85, 185, 285, 385])
        >>> bst.set_value(bst.last_period, np.repeat(0.01, len(bst.last)))
        >>> ezu.adjusted_utility(m, node_cons_eps=bst)
        array([ 9.83391921])

        The last example differs from the rest in that the last values of the `node_cons_eps` will never be
        used. Hence if you want to update the last period consumption, use one of these two methods.

        >>> ezu.adjusted_utility(m, first_period_consadj=0.01)
        array([ 9.84518772])
        """

        utility_tree = BigStorageTree(subinterval_len=self.period_len,
                                      decision_times=self.tree.decision_times)
        cons_tree = BigStorageTree(subinterval_len=self.period_len,
                                   decision_times=self.tree.decision_times)
        ce_tree = BigStorageTree(subinterval_len=self.period_len,
                                 decision_times=self.tree.decision_times)
        cost_tree = SmallStorageTree(decision_times=self.tree.decision_times)

        periods = utility_tree.periods[::-1]
        if period_cons_eps is None:
            period_cons_eps = np.zeros(len(periods))
        if node_cons_eps is None:
            node_cons_eps = BigStorageTree(subinterval_len=self.period_len,
                                           decision_times=self.tree.decision_times)
        self._end_period_utility(m, utility_tree, cons_tree, cost_tree)

        it = self._utility_generator(m, utility_tree, cons_tree, cost_tree,
                                     ce_tree, first_period_consadj)
        i = len(utility_tree)-2
        for u, period in it:
            if period == periods[1]:
                mu_0 = (1.0-self.b) * (u/cons_tree[period])**(1.0-self.r)
                next_term = self.b * (1.0-self.b) / (1.0-self.b*self.growth_term**self.r)
                mu_1 = (u**(1.0-self.r)) * next_term * (cons_tree.last**(self.r-1.0))
                u += (final_cons_eps+period_cons_eps[-1]+node_cons_eps.last) * mu_1
                u +=  (period_cons_eps[i]+node_cons_eps.tree[period]) * mu_0
                utility_tree.set_value(period, u)
            else:
                mu_0, m_1, m_2 = self._period_marginal_utility(period,
                                                               utility_tree,
                                                               cons_tree,
                                                               ce_tree)
                u += (period_cons_eps[i] + node_cons_eps.tree[period])*mu_0
                utility_tree.set_value(period, u)
            i -= 1

        if return_trees:
            return utility_tree, cons_tree, cost_tree, ce_tree

        return utility_tree.tree[0]

    def _period_marginal_utility(self, period, utility_tree, cons_tree, ce_tree):
        """Marginal utility for each node in a period.

        Parameters
        ----------
        period: int
            the current period
        utility_tree: `BigStorageTree` object
            storage tree containing utility values
        cons_tree: `SmallStorageTree` object
            storage tree containing consumption values
        ce_tree: `BigStorageTree` object
            storage tree containing certain equivalence values

        Returns
        -------
        m_0: float
            marginal utility with respect to consumption function
        m_1: float
            marginal utility with respect to consumption next period
        m_2: float
            marginal utility with respect to last period consumption
        """

        damage_period = utility_tree.between_decision_times(period)
        mu_0 = self._mu_0(cons_tree[period], ce_tree[period])

        prev_ce = ce_tree.get_next_period_array(period)
        prev_cons = cons_tree.get_next_period_array(period)
        if utility_tree.is_information_period(period):
            probs = self.tree.get_probs_in_period(damage_period+1)
            up_prob = np.array([probs[i]/(probs[i]+probs[i+1]) for i in range(0, len(probs), 2)])
            down_prob = 1.0 - up_prob

            up_cons = prev_cons[::2]
            down_cons = prev_cons[1::2]
            up_ce = prev_ce[::2]
            down_ce = prev_ce[1::2]

            mu_1 = self._mu_1(cons_tree[period], up_prob, up_cons, down_cons, up_ce, down_ce)
            mu_2 = self._mu_1(cons_tree[period], down_prob, down_cons, up_cons, down_ce, up_ce)
            return mu_0, mu_1, mu_2
        else:
            mu_1 = self._mu_2(cons_tree[period], prev_cons, prev_ce)
            return mu_0, mu_1, None

    def _mu_0(self, cons, ce_term):
        """Marginal utility with respect to consumption function.

        Parameters
        ----------
        cons: float
            consumption value
        ce_term: float
            certain equivalence value

        Returns
        -------
        t1 * t2: float
            the marginal utility w.r.t consumption function.
        """

        t1 = (1.0 - self.b)*cons**(self.r-1.0)
        t2 = (ce_term - (self.b-1.0)*cons**self.r)**((1.0/self.r)-1.0)
        return t1 * t2

    def _mu_1(self, cons, prob, cons_1, cons_2, ce_1, ce_2):
        """ marginal utility with respect to consumption next period.
        Parameters
        ----------
        cons: float
            consumption value
        prob: float
            probability of making move to the next node we're considering
        cons_1, ce_1: float
            consumption/certain equivalence of up-move node
        cons_2, ce_2: float
            consumption/certain equivalence of down-move node

        Returns
        -------
        t1 * t2 * t3 * t5: float
            the marginal utility w.r.t the next period.
        """

        t1 = (1.0-self.b) * self.b * prob * cons_1**(self.r-1.0)
        t2 = (ce_1 - (self.b-1.0) * cons_1**self.r )**((self.a/self.r)-1)
        t3 = (prob * (ce_1 - (self.b*(cons_1**self.r)) + cons_1**self.r)**(self.a/self.r) \
             + (1.0-prob) * (ce_2 - (self.b-1.0) * cons_2**self.r)**(self.a/self.r))**((self.r/self.a)-1.0)
        t4 = prob * (ce_1-self.b * (cons_1**self.r) + cons_1**self.r)**(self.a/self.r) \
             + (1.0-prob) * (ce_2 - self.b * (cons_2**self.r) + cons_2**self.r)**(self.a/self.r)
        t5 = (self.b * t4**(self.r/self.a) - (self.b-1.0) * cons**self.r )**((1.0/self.r)-1.0)

        return t1 * t2 * t3 * t5

    def _mu_2(self, cons, prev_cons, ce_term):
        """Marginal utility with respect to last period consumption.

        Parameters
        ----------
        cons: float
            consumption value at the node
        prev_cons: float
            consumption at the previous node
        ce_term: float
            certain equivalence at the current node

        Returns
        -------
        t1 * t2: float
            marginal utility with respect to the last period of consumption.
        """

        t1 = (1.0-self.b) * self.b * prev_cons**(self.r-1.0)
        t2 = ((1.0 - self.b) * cons**self.r - (self.b - 1.0) * self.b \
             * prev_cons**self.r + self.b * ce_term)**((1.0/self.r)-1.0)
        return t1 * t2

    def partial_grad(self, m, i, delta=1e-8):
        """Calculate the ith element of the gradient vector.

        Parameters
        ----------
        m : ndarray
            array of mitigations
        i : int
            node to calculate partial grad for

        Returns
        -------
        float
            gradient element
        """

        m_copy = m.copy()
        m_copy[i] -= delta
        minus_utility = self.utility(m_copy)
        m_copy[i] += 2*delta
        plus_utility = self.utility(m_copy)
        grad = (plus_utility-minus_utility) / (2*delta)
        return grad
