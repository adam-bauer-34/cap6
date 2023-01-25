"""Climate analysis class.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
4/22/2022

This code contains the ClimateAnalysis class, one of the family of analysis classes in CAP6.
"""

import numpy as np

from ..tools import write_columns_csv, append_to_existing

class ClimateOutput(object):
    """Calculate and save output from the CAP6 model.

    Parameters
    ----------
    utility : `EZUtility` object
        object of utility class

    Attributes
    ----------
    utility : `EZUtility` object
        object of utility class
    prices : ndarray
        SCC prices
    ave_mitigations : ndarray
        average mitigations
    expected_period_price : ndarray
        expected SCC for the period
    expected_period_mitigation : ndarray
        expected mitigation for the period
    expected_period_emissions : ndarray
        expected emission for the period
    """

    def __init__(self, utility):
        self.utility = utility
        self.prices = None
        self.ave_mitigations = None
        self.expected_period_price = None
        self.expected_period_mitigation = None
        self.expected_period_emissions = None
        self.ghg_levels = None

    def calculate_output(self, m):
        """Calculated values based on optimal mitigation. For every **node** the function calculates and saves
            * average mitigation
            * average emission
            * GHG level
            * SCC
        as attributes.

        For every **period** the function also calculates and saves
            * expected SCC/price
            * expected mitigation
            * expected emission
        as attributes.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        """

        emit_baseline = self.utility.damage.emit_baseline
        tree = self.utility.tree
        periods = tree.num_periods

        self.prices = np.zeros(len(m))
        self.ave_mitigations = np.zeros(len(m))
        self.expected_period_price = np.zeros(periods)
        self.expected_period_mitigation = np.zeros(periods)
        self.expected_period_emissions = np.zeros(periods)
        self.ghg_levels = self.utility.damage.ghg_level(m)

        for period in range(0, periods):
            years = tree.decision_times[period]
            period_years = tree.decision_times[period+1] - tree.decision_times[period]
            nodes = tree.get_nodes_in_period(period)
            period_lens = tree.decision_times[:period+1]

            for node in range(nodes[0], nodes[1]+1):
                path = np.array(tree.get_path(node))
                new_m = m[path]
                if node != 0:
                    mean_mitigation = np.dot(new_m, period_lens) / years
                else:
                    mean_mitigation = 0.
                price = self.utility.cost.price(years, m[node], mean_mitigation)
                self.prices[node] = price
                self.ave_mitigations[node] = self.utility.damage.average_mitigation_node(m, node, period)

            probs = tree.get_probs_in_period(period)
            self.expected_period_price[period] = np.dot(self.prices[nodes[0]:nodes[1]+1], probs)
            self.expected_period_mitigation[period] = np.dot(self.ave_mitigations[nodes[0]:nodes[1]+1], probs)

    def save_output(self, m, prefix=None):
        """Function to save calculated values in `calculate_output` in the file `prefix` + 'node_period_output' 
        in the 'data' directory in the current working directory. 

        The function also saves the values calculated in the utility function in the file
        `prefix` + 'tree' in the 'data' directory in the current working directory. 

        If there is no 'data' directory, one is created. 

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        prefix : str, optional
            prefix to be added to file_name

        """
        if prefix is not None:
            prefix += "_"
        else:
            prefix = ""

        #print('in ClimateOutput.save_output(), prefix =',prefix)
        write_columns_csv([m, self.prices, self.ave_mitigations, self.ghg_levels], 
                    prefix+"node_period_output", ["Node", "Mitigation", "Prices", "Average Mitigation",
                    "GHG Level"], [list(range(len(m)))])

        append_to_existing([self.expected_period_price, self.expected_period_mitigation, self.expected_period_emissions],
                           prefix+"node_period_output", header=["Period", "Expected Price", "Expected Mitigation",
                                "Expected Emission"], index=[list(range(self.utility.tree.num_periods))], start_char='\n')


        tree_dict = self.utility.utility(m, return_trees=True)
        self._store_trees(prefix = prefix, tree_dict = tree_dict)

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
        **kwargs
            arbitrary keyword arguments of `BaseStorageTree` objects
        """

        if prefix is None:
            prefix = ""
        for name in tree_dict.keys():
            tree_dict[name].write_columns(prefix + "trees", name, start_year)
