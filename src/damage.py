"""Damage class.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.15.2022

This code contains the abstract class for damages within the CAP6
framework and its subclass which is used in CAP6, DLWDamage.
"""

import sys

import numpy as np
from abc import ABCMeta, abstractmethod

from .damage_simulation import DamageSimulation
from .tools import import_csv

class Damage(object, metaclass=ABCMeta):
    """Abstract damage class for the Climate Asset Pricing 
    model.

    Parameters
    ----------
    tree : `TreeModel` object
        provides the tree structure used
    emit_baseline : `EmissionBaseline` object
        business-as-usual scenario of emissions

    Attributes
    ----------
    tree : `TreeModel` object
        provides the tree structure used
    emit_baseline : `EmissionBaseline` object
        business-as-usual scenario of emissions
    """

    def __init__(self, tree, emit_baseline):
        self.tree = tree
        self.emit_baseline = emit_baseline

    @abstractmethod
    def average_mitigation(self):
        """The average_mitigation function should return a 1D array of the
        average mitigation for every node in the period.
        """
        pass

    @abstractmethod
    def damage_simulation(self):
        """Run the damage simulation.
        """
        pass

    @abstractmethod
    def damage_function(self):
        """The damage_function should return a 1D array of the damages for
        every node in the period.
        """
        pass

class BPWDamage(Damage):
    """Damage class for the EZ-Climate model.

    Provides structure for calculating damages for a given mitigation and
    emission pathway.
    
    Parameters
    ----------
    tree : `TreeModel` object
        provides the tree structure used
    emit_baseline : `EmissionBaseline` object
        business-as-usual scenario of emissions
    climate :  `Climate` object
        provides framework for climate-related calculations
    mitigation_constants: list
        list of constant value mitigation that were used to create
        prototypical emission pathways in damage simulation.

        Example: mitigation_constants = [0.9, 0.6, 0] implies that
        the simulated damage pathways have 90%, 60%, and 0% mitigation.
    draws: int
        number of samples from damage simulation distributions
        (such as TCRE)

    Attributes
    ----------
    tree : `TreeModel` object
        provides the tree structure used
    emit_baseline : `EmissionBaseline` object
        business-as-usual scenario of emissions
    climate: `Climate` object
        provides framework for climate-related calculations
    mitigation_constants: list
        list of constant value mitigation that were used to create
        prototypical emission pathways in damage simulation.

        Example: mitigation_constants = [0.9, 0.6, 0] implies that
        the simulated damage pathways have 90%, 60%, and 0% mitigation.
    draws: int
        number of samples from damage simulation distributions
        (such as TCRE)
    dnum : int
        number of simulated damage paths
    d : ndarray
        simulated damages
    d_rcomb : ndarray
        adjusted simulated damages for recombining tree
    ds: `DamageSimulation` object
        damage simulation object, where simulated damages are created and
        saved. (not initialized until damage_simulation is run.)
    """

    def __init__(self, tree, emit_baseline, climate, mitigation_constants,
                draws):
        super(BPWDamage, self).__init__(tree, emit_baseline)
        self.climate = climate
        self.mitigation_constants = mitigation_constants
        self.draws = draws
        self.dnum = np.shape(mitigation_constants)[0]
        self.d = None
        self.d_rcomb = None
        self.ds = None

    def damage_simulation(self, filename="BPW_simulated_damages.csv",
                          save_simulation=True, dam_func=0, tip_on=True,
                          d_unc=1, t_unc=1):
        """Run damage simulation using `DamageSimulation` object.

        Parameters
        ----------
        filename: string
            name of file to save damage simulation results to
        save_simulation : bool, optional
            True if simulated values should be save, False otherwise. (default
            is True.)
        dam_func: int
            selects which damage function is used in damage simulation.
                0: simulates all three damage functions with equal probability
                (this method incorporates structural uncertainty)
                1: concave down damage function, taken from Burke et al., 2015
                2: concave up damage function, taken from Rose et al., 2017
                3: concave up damage function, taken from Howard & Sterner, 2017
        tip_on: bool
            T/F: are tipping points damages being accounted for? (default
            True.)
        d_unc: int
            toggles parametric uncertainty in damage functions
        t_unc: int
            toggles parametric (TCRE) uncertainty in temperature pathways
        d_var_mult: float
            multiple for the variance in damages (primarily used to test how
            increasing parametric damage uncertainty impacts prices) (default
            is 1.0)
        """

        self.ds = DamageSimulation(tree=self.tree,
                              emission_baseline=self.emit_baseline,
                              climate=self.climate, draws=self.draws,
                              mitigation_constants=self.mitigation_constants,
                              dam_func=dam_func, tip_on=tip_on,
                              d_unc=d_unc, t_unc=t_unc)

        print("Starting damage simulation..")
        self.d = self.ds.simulate(write_to_file=save_simulation,
                             filename=filename)

        print("Done! Now making recombined node structure...")
        self._recombine_nodes()

        print("Damage simulation complete!")

    def _recombine_nodes(self):
        """Create recombining tree damage values.

        Creating damage coefficients for recombining tree. The state reached by
        an up-down move is separate from a down-up move because in general the
        two paths will lead to different degrees of mitigation and therefore of
        GHG level. A 'recombining' tree is one in which the movement from one
        state to the next through time is nonetheless such that an up move
        followed by a down move leads to the same fragility.
        """

        nperiods = self.tree.num_periods
        sum_class = np.zeros(nperiods, dtype=int)
        new_state = np.zeros([nperiods, self.tree.num_final_states], dtype=int)
        temp_prob = self.tree.final_states_prob.copy()
        self.d_rcomb = self.d.copy()

        # make a list of the number of nodes in each final grouping. for a N =
        # 6 tree, this is [1, 5, 10, 10, 5, 1].
        for old_state in range(self.tree.num_final_states):
            temp = old_state
            n = nperiods-2
            d_class = 0
            while n >= 0:
                if temp >= 2**n:
                    temp -= 2**n
                    d_class += 1
                n -= 1
            sum_class[d_class] += 1
            # NOTE: this was the old way of doing things, which is wrong and
            # mislables nodes.
            #new_state[d_class, sum_class[d_class]-1] = old_state

        # assign node states to the groupings made above. these are simply
        # incremental from 0. low is the lower bound of the range of node
        # states, up is the upper bound, new_state is the revised state of the
        # path dependent tree in the recombined tree.
        for i in range(len(sum_class)):
            low = np.sum(sum_class[:i])
            up = np.sum(sum_class[:i+1])
            new_state[i, :sum_class[i]] = np.arange(low, up, 1, dtype=int)

        # make probability of reaching a set of nodes in the final period; this
        # is different than reaching a single node in the final period. more
        # nodes = higher probability of reaching them, so their damages are
        # weighted higher.
        sum_nodes = np.append(0, sum_class.cumsum())
        prob_sum =\
        np.array([self.tree.final_states_prob[sum_nodes[i]:sum_nodes[i+1]].sum()\
                  for i in range(len(sum_nodes)-1)])

        # reassign damages - this is where the "recombined" damages are
        # created.
        for period in range(nperiods):
            for k in range(self.dnum):
                d_sum = np.zeros(nperiods)
                old_state = 0
                for d_class in range(nperiods):
                    test_1 = self.tree.final_states_prob[old_state:old_state+sum_class[d_class]]
                    test_2 = self.d_rcomb[k, old_state:old_state+sum_class[d_class], period]
                    d_sum[d_class] = (test_1 * test_2).sum()
                    old_state += sum_class[d_class]
                    self.tree.final_states_prob[new_state[d_class, 0:sum_class[d_class]]] = temp_prob[0]

                for d_class in range(nperiods):
                    self.d_rcomb[k, new_state[d_class, 0:sum_class[d_class]],\
                                 period] = d_sum[d_class] / prob_sum[d_class]

        self.tree.node_prob[-len(self.tree.final_states_prob):] = self.tree.final_states_prob

        # change around probabilities in tree to align with recombined
        # structure
        for p in range(1,nperiods-1):
            nodes = self.tree.get_nodes_in_period(p)
            for node in range(nodes[0], nodes[1]+1):
                end_states = self.tree.reachable_end_states(node)
                self.tree.node_prob[node] = self.tree.final_states_prob[end_states].sum()

    def import_damages(self, file_name="simulated_damages_BPW_TCRE"):
        """Import saved simulated damages.

        File must be saved in 'data' directory
        inside current working directory. Save imported values in `d`.

        Parameters
        ----------
        file_name : str, optional
            name of file of saved simulated damages

        Raises
        ------
        IOError
            If file does not exist.
        """

        try:
            d = import_csv(file_name, ignore="#", header=False)
        except IOError as e:
            print(("Could not import simulated damages:\n\t{}".format(e)))
            sys.exit(0)

        n = self.tree.num_final_states
        self.d = np.array([d[n*i:n*(i+1)] for i in range(0, self.dnum)],
                          dtype=object)
        self._recombine_nodes()
        print("Damages imported successfully!")

    def damage_function(self, m, period, is_last=False):
        """Calculate the damage for every node in a period, based on mitigation
        actions `m`.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        period : int
            period to calculate damages for
        is_last: bool
            is it the last period? (default: False)

        Returns
        -------
        damages: ndarray
            array of damages at each node
        """

        nodes = self.tree.get_num_nodes_period(period)
        damages = np.zeros(nodes)
        for i in range(nodes):
            node = self.tree.get_node(period, i)
            damages[i] = self._damage_function_node(m, node, is_last=is_last)
        return damages

    def _damage_function_node(self, m, node, is_last=False):
        """Calculate the damage at a node.

        Use the recombined tree damage values to calculate the damage
        at any node, for any mitigation vector, using interpolation between the
        three recombined damage trees. (Recall, these were made out of the
        output of the damage simulation, so there are three of them -- one for
        each mitigation constant value.) The interpolation between the lowest
        damage (highest mitigation) and the middle damage (middle mitigation)
        is prescribed with a quadratic interpolation. The interpolation is
        linear to the highest possible value.

        The damage value returned has two components: one from the recombined
        damage trees, and the other from the "penalty function," which
        penalizes an individual from mitigated CO2 in the atmosphere beneath
        preindustrial values. (Such actions would potentially have disatorous
        outcomes; if you thought global warming was bad, imagine global
        cooling!)

        Parameters
        ----------
        m: nd array
            mitigation vector up to the node

        node: int
            node that we're calculating the damage for

        is_last: bool
            is it the last period?

        Returns
        -------
        total_damage: float
            total damage at the node
        """

        # no damages at first node.
        if node == 0:
            return 0.0

        # find what period we're in 
        # (i.e., how many decisions have already been made?)
        period = self.tree.get_period(node)

        if is_last:
            period += 1

        # get mitigated baseline for current node
        indiv_mit_emissions_cumemit, _ = self.emit_baseline.get_mitigated_baseline(m=m,
                                                                        node=node,
                                                                        baseline='cumemit',
                                                                        is_last=is_last)

        # calc ghg content at node using carbon cycle model
        node_indiv_ghg_level = self.climate.get_conc_at_node(m=m, node=node,
                                                             is_last=is_last)

        # note cumemit at current time
        node_indiv_cumemit = indiv_mit_emissions_cumemit[-1]

        # get indiviudal damage at current node
        node_indiv_damage = self._get_interp_damage(node_indiv_cumemit, node,
                                                    period)

        # penalize individuals for overmitigating. functional form taken from
        # Declining CO2 price paths, Daniel et al. 2017
        damage_penalty_function = 1.0 / (1 + np.exp(0.05*(node_indiv_ghg_level-200)))

        # return total damage
        return node_indiv_damage + damage_penalty_function

    def _get_interp_damage(self, cemit, node, period):
        """Get interpolated damages at node.

        This function uses the recombined damage values to find the economic
        damages at the current node. We use a linear interpolation between
        damage values for different prototypical damage trees.

        Parameters
        ----------
        cemit: float
            cumulative emissions at current node

        node: int
            current node

        period: int
            current period

        Returns
        -------
        damage: float
            expected value of damage at current node
        """

        # note current cumulative emissions at the current period for every
        # mitigation constant value
        mit_cumemit_per = np.array([((1 - self.mitigation_constants[i]) *
                           self.emit_baseline.baseline_cumemit_periods)[period]
                           for i in range(self.dnum)])

        # find end states to get probabilities for weighted sum
        end_states = self.tree.reachable_end_states(node)
        probs = self.tree.final_states_prob[end_states]

        # create damage to interpolate to for each mitigation value by carrying out a 
        # weighted sum
        # NOTE: period - 1 is used because there are no simulated damage values
        # for the period = 0. So we shift accordingly.
        d_per = [(probs * self.d_rcomb[i, end_states,
                                       period-1]).sum()/probs.sum() for i in
                 range(self.dnum)]

        # find indexes where current cumulative emissions are greater than
        # cumulative emissions in the prototypical emissions pathways.
        # sometimes the algorithm has a negative cumulative emissions (due to
        # early overmitigation) or too much (due to under mitigation). In those
        # cases, we return zero damage for overmitigating and the maximum
        # damage for under.
        try:
            c_ind = np.where(mit_cumemit_per < cemit)[0][-1]
        except IndexError:
            if cemit <= mit_cumemit_per[0]:
                #print("Exception triggered, returning 0 damage.")
                return 0.0
            elif cemit >= mit_cumemit_per[-1]:
                #print("Exception triggered, returning max damage.")
                return d_per[-1]

        # if c_ind is not defined (i.e., something went wrong above) raise the
        # error and print out the results for troubleshooting
        try:
            cemit_less = mit_cumemit_per[c_ind]
        except UnboundLocalError:
            print(d_per, mit_cumemit_per, cemit)
            raise

        # if c_ind = self.dnum - 1, that means that the cumulative emissions
        # to this period are equal the max allowable. so there is no "next"
        # cumulative emissions -- we're at the max amount. if this is the case,
        # we just return the maximum damages.
        try:
            cemit_great = mit_cumemit_per[c_ind + 1]
        except IndexError:
            if c_ind == self.dnum-1:
                return d_per[-1]
            elif c_ind == 0:
                return 0.0

        # note damages corresponding to these cumulative emissions values
        d_less = d_per[c_ind]
        d_great = d_per[c_ind + 1]

        # find slope of linear interpolation line
        slope = (d_great - d_less) * (cemit_great - cemit_less)**(-1)

        # interpolate using the usual linear formula
        damage = slope * (cemit - cemit_less) + d_less
        return damage

    def average_mitigation(self, m, period, is_last=False):
        """Calculate the average mitigation for all nodes in a period.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        period : int
            period to calculate average mitigation for
        is_last: bool
            is it the last period?

        Returns
        -------
        ave_mitigation: ndarray
            average mitigations
        """

        nodes = self.tree.get_num_nodes_period(period)
        ave_mitigation = np.zeros(nodes)
        for i in range(nodes):
            node = self.tree.get_node(period, i)
            ave_mitigation[i] = self.average_mitigation_node(m, node, period,
                                                             is_last)
        return ave_mitigation

    def average_mitigation_node(self, m, node, period=None, is_last=False):
        """Calculate the average mitigation until node.

        Technically, this is the weighted average of mitigation, where the
        emissions baseline is the "weight."

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        node : int
            node for which average mitigation is to be calculated for
        period : int, optional
            the period the node is in
        is_last: bool
            is it the last period?

        Returns
        -------
        float
            average mitigation
        """

        if period == 0:
            return 0
        if period is None:
            period = self.tree.get_period(node)

        mit_cumemit, _ = self.emit_baseline.get_mitigated_baseline(m, node,
                                                                baseline='cumemit',
                                                                  is_last=is_last)
        period_ind = self.emit_baseline.dec_times_ind[period] + 1
        baseline_up_to_period = self.emit_baseline.baseline_cumemit[:period_ind]

        # calculate the ratio of mitigated cumulative emissions to total
        # cumulative emissions up to and including the current node, while
        # subtracting off the cumulative emissions already present before
        # optimization.
        average_mitigation_node = (mit_cumemit[-1]\
                                - self.emit_baseline.CUMEMIT_2019)\
                                * (baseline_up_to_period[-1]\
                                - self.emit_baseline.CUMEMIT_2019)**(-1)
        return 1 - average_mitigation_node

    def ghg_level(self, m, periods=None):
        """Calculate the GHG levels for more than one period.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        periods : int, optional
            number of periods to calculate GHG levels for

        Returns
        -------
        ndarray
            GHG levels
        """

        if periods is None:
            periods = self.tree.num_periods-1
        if periods >= self.tree.num_periods:
            ghg_level = np.zeros(self.tree.num_decision_nodes+self.tree.num_final_states)
        else:
            ghg_level = np.zeros(self.tree.num_decision_nodes)
        for period in range(periods+1):
            start_node, end_node = self.tree.get_nodes_in_period(period)
            if period >= self.tree.num_periods:
                add = end_node-start_node+1
                start_node += add
                end_node += add
            nodes = np.array(list(range(start_node, end_node+1)))
            ghg_level[nodes] = self.ghg_level_period(m, nodes=nodes)
        return ghg_level

    def ghg_level_period(self, m, period=None, nodes=None):
        """Calculate the GHG levels corresponding to the given mitigation.
        Need to provide either `period` or `nodes`.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        period : int, optional
            what period to calculate GHG levels for
        nodes : ndarray or list, optional
            the nodes to calculate GHG levels for

        Returns
        -------
        ndarray
            GHG levels
        """

        if nodes is None and period is not None:
            start_node, end_node = self.tree.get_nodes_in_period(period)
            if period >= self.tree.num_periods:
                add = end_node-start_node+1
                start_node += add
                end_node += add
            nodes = np.array(list(range(start_node, end_node+1)))
        if period is None and nodes is None:
            raise ValueError("Need to give function either nodes or the period")

        ghg_level = np.zeros(len(nodes))
        for i in range(len(nodes)):
            ghg_level[i] = self._ghg_level_node(m, nodes[i])
        return ghg_level

    def _ghg_level_node(self, m, node):
        """Calcs CO2 concentrations at a node.

        Makes mitigated BAU pathway to particular node, then calls
        climate.get_conc_at_node to calculate the concentrations in ppm.

        Parameters
        ----------
        m: nd array
            Array of mitigation values
        node: int
            Node we're calculating CO2 concentrations for

        Returns
        -------
        conc_node: float
            CO2 concentrations at the node
        """

        conc_node = self.climate.get_conc_at_node(m=m, node=node)
        return conc_node
