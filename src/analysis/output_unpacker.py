"""Output unpacker class.

This class makes a series of quantities of interest from the output files of a
run of CAP6. These can be used in the analysis Jupyter notebook to make
figures to better understand model output and model inefficiencies.
"""

import numpy as np
import pandas as pd

class OutputUnpacker:
    """Output Unpacker class.

    Class which contains various attributes and methods to represent the
    approximate output of CAP6.

    Parameters
    ----------
    filename: string
        name of file
    description: string
        file description, usually is given by the name of the model run
    _type: string
        "pickle" or "output", tells the class the type of the file of interest
    tree: `TreeModel` object
        tree model used in the CAP6 model run
    emit_baseline: `EmissionBaseline` object
        emission baseline used in model run
    climate: `Climate` object
        climate representation used in model run
    damage: `Damage` object
        damage class used in model run

    """
    def __init__(self, filename, description, _type, tree, emit_baseline,
                 climate, damage):
        self.filename = filename
        self.description = description
        self._type = _type
        self.tree = tree
        self.emit_baseline = emit_baseline
        self.climate = climate
        self.damage = damage

        # make some empty arrays for output quantities
        # econ damages in node and path notation
        self.econ_dam_node = np.zeros(self.tree.num_decision_nodes)
        self.econ_dam_path = np.zeros((self.tree.num_final_states,
                                       self.tree.num_periods))

        self.exp_mit_node = np.zeros(self.tree.num_decision_nodes)

        # mitigated emission and temperature pathway in path notation as well
        # as full time series
        self.emis_gtco2_path = np.zeros((self.tree.num_final_states,
                                   self.tree.num_periods))
        self.temp_path = np.zeros((self.tree.num_final_states,
                                   self.tree.num_periods))

        # make full time series of emissions and temperature. we take ten off
        # of the total number of points because the system is uninteresting
        # past 2300; emissions are turned off, so temperature and emissions
        # aren't changing.
        self.temp_full = np.zeros((self.tree.num_final_states,
                                   np.shape(self.emit_baseline.times)[0] - 5))
        self.emis_gtco2_full = np.zeros((self.tree.num_final_states,
                                   np.shape(self.emit_baseline.times)[0] - 5))

        if self._type == 'pickle':
            # read picklefile
            self.pickle_file = pd.read_pickle(self.filename)

            # extract parameters from file
            self.m_opt_node = self.pickle_file['m_opt']
            self.d_rcomb = self.pickle_file['df.d_rcomb']

            # make average temperature pathways and economic damage pathways
            # using extracted file values
            self._make_temp_emis_path() # make temperature and emissions pathways
            self._make_econ_damages() # make economic damages

            # transfer node notation arrays to path notation
            self.m_opt_path = self._get_node_to_path(self.m_opt_node)
            self.econ_dam_path = self._get_node_to_path(self.econ_dam_node)

        elif self._type == 'output':
            # extract model output; col = 1 is mitigation, col = 2 is price,
            # col = 5 is ghg concentrations
            self.output_data = np.genfromtxt(self.filename, delimiter=';',
                                             usecols=(1,2,5), skip_header=1,
                                             skip_footer=9)

            # make model output into their own arrays
            self.m_opt_node = self.output_data[:, 0]
            self.price_node = self.output_data[:, 1]
            self.ghg_lvl_node = self.output_data[:, 2]

            # make expected mitigation
            self._make_expected_mit()

            # turn node notation quantities above into path notation
            self.m_opt_path = self._get_node_to_path(self.m_opt_node)
            self.price_path = self._get_node_to_path(self.price_node)
            self.ghg_lvl_path = self._get_node_to_path(self.ghg_lvl_node)
            self.exp_mit_path = self._get_node_to_path(self.exp_mit_node)

            # make average temperature pathways using model output
            self._make_temp_emis_path() # make temperature and emissions pathways

    def _make_temp_emis_path(self):
        """Make temperature and mitigated emissions pathways.

        This function takes the optimal mitigation vector and uses it to
        compute the mitigated emissions pathways (in GtCO2/year) and the
        temperature (in K) for every path through the tree.
        """

        period_inds = self.emit_baseline.dec_times_ind
        # loop from 31 -> 63
        for node in range(self.tree.num_decision_nodes -
                          self.tree.num_final_states,
                          self.tree.num_decision_nodes):

            ind = node - 31

            self.emis_gtco2_full[ind, :], _ =\
            self.emit_baseline.get_mitigated_baseline(self.m_opt_node,
                                                      node=node,
                                                      baseline='gtco2')
            self.emis_gtco2_path[ind, :] = self.emis_gtco2_full[ind,
                                                                period_inds[:-1]]
            mit_cumemit, _  =\
            self.emit_baseline.get_mitigated_baseline(self.m_opt_node,
                                                      node=node,
                                                      baseline='cumemit')

            self.temp_full[ind, :] = mit_cumemit\
                                     * self.climate.TCRE_BEST_ESTIMATE
            self.temp_path[ind, :] = self.temp_full[ind, period_inds[:-1]]

    def _make_econ_damages(self):
        """Make economic damages at every node.

        This function takes the optimal mitigation vector and computes the
        damages at every node using the Damage class function
        _damage_function_node.
        """

        for node in range(0, self.tree.num_decision_nodes):
            self.econ_dam_node[node] = self.damage._damage_function_node(self.m_opt_node, node=node)

    def _make_expected_mit(self):
        """Make expected mitigation at every node.

        This function takes the optimal mitigation vector and computes the
        weighted average of mitigation.
        """

        for node in range(self.tree.num_decision_nodes):
            self.exp_mit_node[node] = self.damage.average_mitigation_node(self.m_opt_node, node)

    def _get_node_to_path(self, array):
        """Take an array in node notation and create an array in path notation.

        In CAP6, there are two ways to store output: in node notation
        or in path notation. This code takes an array in node notation (i.e.,
        an array with shape (tree.num_decision_nodes,)) and transforms it into
        an array in path notation (i.e., an array with shape (tree.num_periods,
        tree.num_final_states)).

        Parameters
        ----------
        array: (tree.num_decision_nodes,) array
            array in node notation

        Returns
        -------
        path_array: (tree.num_final_states, len(tree.decision_times)) array
            array in path notation
        """

        path_array = np.zeros((self.tree.num_final_states,
                              self.tree.num_periods))

        for node in range(self.tree.num_decision_nodes\
                          - self.tree.num_final_states,
                          self.tree.num_decision_nodes):
            tmp_path = self.tree.get_path(node)
            path_array[node - 31, :] = array[tmp_path]
        return path_array

