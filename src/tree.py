"""Tree Model class for CAP6.

Adam M. Bauer
University of Illinios at Urbana Champaign
adammb4@illinois.edu
6.16.2022

This code contains the Tree Model class for CAP6. It contains numerous
useful attributes and methods relating to the tree model, which is at the
foundation of CAP6's financial model for CO2 price.
"""

import numpy as np
import itertools as it

class TreeModel(object):
    """Tree model for CAP6. It provides the structure of a
    non-recombining tree.

    Parameters
    ----------
    decision_times : ndarray or list
        years in the future where decisions will be made
    prob_scale : float, optional
        scaling constant for probabilities

    Attributes
    ----------
    decision_times : ndarray
        years in the future where decisions will be made
    prob_scale : float
        scaling constant for probabilities
    node_prob : ndarray
        probability of reaching node from period 0
    final_states_prob : ndarray
        last periods `node_prob`
    paths: nd array
        matrix of paths through the tree
    """

    def __init__(self, decision_times, prob_scale=1.0):
        self.decision_times = decision_times
        if isinstance(self.decision_times, list):
            self.decision_times = np.array(self.decision_times)
        self.prob_scale = prob_scale
        self.node_prob = None
        self.final_states_prob = None
        self.paths = None
        self._create_probs()
        self._create_paths()

    @property
    def num_periods(self):
        """int: the number of periods in the tree.
        """

        return len(self.decision_times)-1

    @property
    def num_decision_nodes(self):
        """int: the number of nodes in tree.
        """

        return (2**self.num_periods) - 1

    @property
    def num_final_states(self):
        """int: the number of nodes in the last period.
        """

        return 2**(self.num_periods-1)

    def _create_probs(self):
        """Creates the probabilities of every nodes in the tree structure.
        """

        self.final_states_prob = np.zeros(self.num_final_states)
        self.node_prob = np.zeros(self.num_decision_nodes)
        self.final_states_prob[0] = 1.0
        sum_probs = 1.0
        next_prob = 1.0

        for n in range(1, self.num_final_states):
            next_prob = next_prob * self.prob_scale**(1.0 / n)
            self.final_states_prob[n] = next_prob
        self.final_states_prob /= np.sum(self.final_states_prob)

        self.node_prob[self.num_final_states-1:] = self.final_states_prob
        for period in range(self.num_periods-2, -1, -1):
            for state in range(0, 2**period):
                pos = self.get_node(period, state)
                self.node_prob[pos] = self.node_prob[2*pos + 1]\
                                    + self.node_prob[2*pos + 2]

    def _create_paths(self):
        """Create a matrix of all possible paths through the tree.

        This function creates a (2^(p-1), p) matrix of paths through the
        underlying tree on CAP6, where p is the number of periods. This
        matrix is vital to properly calculating damages, mitigated emissions
        pathways, among other quantities.

        We do this in three parts. We first make a matrix of all possible
        decision vectors through the tree (that is, vectors of ones and zeros,
        where ones are steps up the tree and steps down are zero). We then
        order these paths by their fragility, and map the sorted paths in terms
        of decisions (ones and zeros) to node numbers.
        """

        self.paths = np.zeros((2**(self.num_periods-1), self.num_periods),
                              dtype=np.int)

        # make decision matrix
        dec_matrix = self._get_decision_matrix()

        # sort dec matrix by values of fragility (highest to lowest)
        sorted_dec_matrix = self._sort_dec_matrix(dec_matrix)

        # now we map the sorted set of decision matrix paths to node number
        # paths. the first step is to generate a set of node number modifiers
        # that are important for the node number calculation.
        node_num_modifiers = self._generate_modifiers()

        # now loop through and make paths recursively. node number paths all
        # start at node zero, and the remaining node numbers can be found
        # using:
        #   n_i = n_{i-1} + 2^{i-1} + modifier[i, D],
        # where D is the number of down steps (zeros) taken up until that node
        # in the path.
        path_number = 0
        for path in sorted_dec_matrix:
            for i in range(1, len(path) + 1):
                S = np.sum(path[:i])
                D = int(len(path[:i]) - S)
                self.paths[path_number, i] = self.paths[path_number, i - 1]\
                                           + 2**(i-1)\
                                           + node_num_modifiers[i, D]
            path_number += 1

    def _get_decision_matrix(self):
        """Create the set of all possible paths through the tree in terms of
        ones and zeros; these are called the "decision vectors."

        Returns
        -------
        dec_matrix: nd array
            the matrix of all possible combinations of 1s and 0s
        """

        dec_matrix = np.zeros((2**(self.num_periods - 1),
                               self.num_periods - 1))
        tracker = 1

        # loop through and make each decision vector combinatorically, leaving
        # the first row of dec_matrix alone as (0,0,0...,0) is a valid path
        for cluster in range(1, self.num_periods):
            tmp_which = np.array(list(it.combinations(range(self.num_periods -
                                                            1), cluster)))
            tmp_grid = np.zeros((len(tmp_which), self.num_periods - 1),
                                dtype=np.int)

            tmp_grid[np.arange(len(tmp_which))[None].T, tmp_which] = 1
            rows = len(tmp_grid)
            dec_matrix[tracker:tracker+rows] = tmp_grid
            tracker += rows

        return dec_matrix

    def _sort_dec_matrix(self, dec_matrix):
        """Sort the decision matrix rows by values of their fragility.

        The highest fragility paths are placed at the top of the matrix, while
        the lowest are at the bottom. I.e., the path (1,1,1,...,1) is the
        highest fragility while the path (0,0,0,...,0) is the lowest.

        Parameters
        ----------
        dec_matrix: nd array
            matrix of decision vectors (i.e., vectors of ones and zeros)

        Returns
        -------
        sorted_dec_matrix: nd array
            dec_matrix vectors sorted by their fragility
        """

        fragilities = []

        # go through paths in decision matrix and find fragility
        for path in dec_matrix:
            fragility = self._get_fragility(path)
            fragilities.append(fragility)

        sorted_dec_matrix_inds = np.argsort(fragilities)[::-1]
        sorted_dec_matrix = dec_matrix[sorted_dec_matrix_inds, :]
        return sorted_dec_matrix

    def _get_fragility(self, path):
        """Calculate fragility.

        The fragility is here represented by, for a given decision vector x:

            fragility = \sum_j (x_j (2^len(x) - 2^x_j))

        NOTE: Our way of calculating the fragility here is not unique. It is
        just (a) qualitatively correct and (b) computationally efficient.

        Parameters
        ----------
        path: nd array
            the decision vector in question

        Returns
        -------
        fragility: float
            the fragility
        """

        inds = np.asarray([i for i in range(len(path))])
        weight = 2**len(path) - 2**inds
        fragility = np.sum(path * weight)
        return fragility

    def _generate_modifiers(self):
        """Generate node number calculation modifiers.

        These modifiers change by period and by the number of down steps one
        takes along a given path. There are five main rules they follow. If the
        set of modifiers in a given period for some number of down steps is is
        F_p(D), then we have:
            - F_p(D = 0) = 0
            - F_p(D = 1) = 1
            - |F_p(D)| = p + 1
            - F_p(D = p) = 2**(p-1)
            - F_p(D) = F_{p-1}(D) + F_{p-1}(D-1)

        Returns
        -------
        modifiers: nd array
            matrix of modifiers as described above
        """

        modifiers = np.zeros((self.num_periods, self.num_periods),
                              dtype=np.int)

        # set rule F_p(D = 1) = 1
        modifiers[1:, 1] = 1

        # set diagonals rule F_p(D = p) = 2**(p-1)
        for per in range(1, self.num_periods):
            modifiers[per, per] = 2**(per - 1)

        # fill in rest of values using rule: F_p(D) = F_{p-1}(D) + F_{p-1}(D-1)
        for per in range(3, self.num_periods):
            for i in range(2, self.num_periods - 1):
                if i == per:
                    continue
                else:
                    modifiers[per, i] = modifiers[per - 1, i]\
                                      + modifiers[per - 1, i - 1]
        return modifiers

    def get_num_nodes_period(self, period):
        """Returns the number of nodes in the period.

        Parameters
        ----------
        period : int
            period

        Returns
        -------
        int
            number of nodes in period

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_num_nodes_period(2)
        4
        >>> t.get_num_nodes_period(5)
        32
        """

        if period >= self.num_periods:
            return 2**(self.num_periods-1)
        return 2**period

    def get_nodes_in_period(self, period):
        """Returns the first and last nodes in the period.

        Parameters
        ----------
        period : int
            period

        Returns
        -------
        int
            number of nodes in period

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_nodes_in_period(0)
        (0, 0)
        >>> t.get_nodes_in_period(1)
        (1, 2)
        >>> t.get_nodes_in_period(4)
        (15, 30)
        """

        if period >= self.num_periods:
            period = self.num_periods-1
        nodes = self.get_num_nodes_period(period)
        first_node = self.get_node(period, 0)
        return (first_node, first_node+nodes-1)

    def get_node(self, period, state):
        """Returns the node in period and state provided.

        Parameters
        ----------
        period : int
            period
        state : int
            node number relative to period, with zero corresponding to the node
            with highest fragility in the period, and 2^period corresponding to
            the node with the least fragility.

        Returns
        -------
        int
            node number relative to the entire tree

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_node(1, 1)
        2
        >>> t.get_node(4, 10)
        25
        >>> t.get_node(4, 20)
        ValueError: No such state in period 4

        Raises
        ------
        ValueError
            If period is too large or if the state is too large
            for the period.
        """

        if period > self.num_periods:
            raise ValueError("Given period is larger than number of periods")
        if state >= 2**period:
            raise ValueError("No such state in period {}".format(period))
        if period == self.num_periods:
            period -= 1
        return 2**period + state - 1

    def get_state(self, node, period=None):
        """Returns the state the node represents.

        Parameters
        ----------
        node : int
            the node
        period : int, optional
            the period

        Returns
        -------
        int
            state

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_state(0)
        0
        >>> t.get_state(4, 2)
        1

        """
        if node >= self.num_decision_nodes:
            return node - self.num_decision_nodes
        if not period:
            period = self.get_period(node)
        return node - (2**period - 1)

    def get_period(self, node):
        """Returns what period the node is in.

        Parameters
        ----------
        node : int
            the node

        Returns
        -------
        int
            period

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_period(0)
        0
        >>> t.get_period(4)
        2

        """
        if node >= self.num_decision_nodes:
            raise ValueError("Passed node number exceeds total nodes within\
                             the model.")

        for i in range(0, self.num_periods):
            if int((node+1) / 2**i) == 1:
                return i

    def get_parent_node(self, child):
        """Returns the previous or parent node of the given child node.

        Parameters
        ----------
        child : int
            the child node

        Returns
        -------
        int
            partent node

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_parent_node(2)
        0
        >>> t.get_parent_node(4)
        1
        >>> t.get_parent_node(10)
        4
        """

        if child == 0:
            return 0
        if child > self.num_decision_nodes:
            return child - self.num_final_states
        if child % 2 == 0:
            return int((child - 2) / 2)
        else:
            return int((child - 1 ) / 2)

    def get_path(self, node):
        """Returns the unique path taken to come to given node.

        Parameters
        ----------
        node : int
            the node

        Returns
        -------
        ndarray
            path to get to `node`

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_path(2)
        array([0, 2])
        """

        paths_ind, loc_ind = np.where(self.paths == int(node))
        path = self.paths[paths_ind[0], :loc_ind[0]+1]
        return path

    def get_probs_in_period(self, period):
        """Returns the probabilities to get from period 0 to nodes in period.

        Parameters
        ----------
        period : int
            the period

        Returns
        -------
        ndarray
            probabilities

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.get_probs_in_period(2)
        array([ 0.25,  0.25,  0.25,  0.25])
        >>> t.get_probs_in_period(4)
        array([ 0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
                0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
                0.0625,  0.0625])
        """

        first, last = self.get_nodes_in_period(period)
        return self.node_prob[list(range(first, last+1))]

    def reachable_end_states(self, node):
        """Returns what future end states can be reached from given node.

        Parameters
        ----------
        node : int
            the node

        Returns
        -------
        end_states: nd array
            reachable end states from the node

        Examples
        --------
        >>> t = TreeModel([0, 15, 45, 85, 185, 285, 385])
        >>> t.reachable_end_states(0)
        np.array([0, 1, ..., self.num_final_states - 1])
        >>> t.reachable_end_states(10)
        np.array([5, 9, 12, 18])
        >>> t.reachable_end_states(32)
        np.array([1])
        """

        paths_ind, _ = np.where(self.paths == int(node))
        end_states = self.paths[paths_ind].T[-1] -  self.num_final_states + 1
        return end_states
