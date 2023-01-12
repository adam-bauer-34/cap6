"""Emission baseline class.

Adam M. Bauer
adammb4@illinois.edu
University of Illinois at Urbana Champaign
3.24.2022

Contains the abstract class BusinessAsUsual as well as its subclass,
BPWBusinessAsUsual, which is implemented in TCREZClimate.
"""

import numpy as np
from abc import ABC, abstractmethod

from .tools import get_integral_var_ub, import_csv

class EmissionBaseline(ABC):
    """Abstract emissions baseline class for TCREZClimate.

    Attributes
    ----------
    times: nd array
        times (in years) that the emissions are taking place

    baseline_gtco2: nd array
        baseline emissions in GtCO2/year

    baseline_ppm: nd array
        baseline emissions in ppm CO2/year

    baseline_cumemit: nd array
        cumulative emissions of baseline in 1000 GtCO2

    baseline_gtco2_periods: nd array
        baseline emissions in GtCO2/year evaluated at tree.decision_times

    baseline_ppm_periods: nd array
        baseline emissions in ppm CO2/year evaluated at tree.decision_times

    baseline_cumemit_periods: ndarray
        cumulative emissions in 1000 GtCO2 evaluated at tree.decision_times

    dec_times_ind: ndarray
        indexes at which decision times are within self.times

    DELTA_T: int
        time difference between emission data points

    CUMEMIT_2019: float
        cumulative emissions as of 2019 (in 1000 GtCO2)

    GTCO2_TO_PPM: float
        conversion factor which takes GtCO2 and results in ppm CO2

    Methods
    -------
    baseline_emission_setup:
        sets up various class attributes

    get_mitigated_baseline:
        for a vector or value of mitigation, return mitigated baseline
    """

    def __init__(self):
        self.times = None
        self.baseline_gtco2 = None
        self.baseline_ppm = None
        self.baseline_cumemit = None
        self.baseline_gtco2_periods = None
        self.baseline_ppm_periods = None
        self.baseline_cumemit_periods = None
        self.dec_times_ind = None
        self.DELTA_T = None
        self.CUMEMIT_2019 = 2.39 # 1000 GtCO2
        self.GTCO2_TO_PPM = 7.8**(-1) # takes GtCO2 -> ppm CO2

    @abstractmethod
    def baseline_emission_setup(self):
        pass

    @abstractmethod
    def get_mitigated_baseline(self):
        pass

class BPWEmissionBaseline(EmissionBaseline):
    """Baseline CO2 emission pathways.

    This class contains information on the various emissions pathways
    used by TCREZClimate. It features baseline flexibility, allowing the code
    to change which SSP baseline we use with a parameter.

    Parameters
    ----------
    tree: `TreeModel` object
        tree structure of the model

    baseline_num: int
        baseline number, such that:
            1: SSP1
            2: SSP2
            3: SSP3
            4: SSP4
            5: SSP5
        an error will be raised if any number other that
        1-5 are given.

    Attributes
    ----------
    times: nd array
        times (in years) that the emissions are taking place

    baseline_gtco2: nd array
        baseline emissions in GtCO2/year

    baseline_ppm: nd array
        baseline emissions in ppm CO2/year

    baseline_cumemit: nd array
        cumulative emissions of baseline in 1000 GtCO2

    baseline_gtco2_periods: nd array
        baseline emissions in GtCO2/year evaluated at tree.decision_times

    baseline_ppm_periods: nd array
        baseline emissions in ppm CO2/year evaluated at tree.decision_times

    baseline_cumemit_periods: ndarray
        cumulative emissions in 1000 GtCO2 evaluated at tree.decision_times

    dec_times_ind: nd array
        indexes at which decision times are within self.times

    DELTA_T: int
        time difference between emission data points

    CUMEMIT_2019: float
        cumulative emissions as of 2019 (in 1000 GtCO2)

    GTCO2_TO_PPM: float
        conversion factor which takes GtCO2 and results in ppm CO2

    Methods
    -------
    baseline_emission_setup:
        sets up various class attributes

    _make_extension:
        per Meinshausen et al., 2020 (https://doi.org/10.5194/gmd-13-3571-2020)
        the extensions of the various SSPs are just (basically, going off of
        Figure 2) linearly connect the final emissione value in 2100 to zero in
        2250. this method carries out this prescription and makes the final
        emissions time series.

    get_mitigated_baseline:
        for a given node and mitigation vector/value, make the mitigated version
        of a given baseline.
    """

    def __init__(self, tree, baseline_num):
        self.tree = tree
        self.baseline_num = baseline_num
        self.times = None
        self.baseline_gtco2 = None
        self.baseline_ppm = None
        self.baseline_cumemit = None
        self.baseline_gtco2_periods = None
        self.baseline_ppm_periods = None
        self.baseline_cumemit_periods = None
        self.dec_times_ind = None
        self.DELTA_T = None
        self.CUMEMIT_2019 = 2.39 # 1000 GtCO2
        self.GTCO2_TO_PPM = 7.8**(-1) # takes GtCO2 -> ppm CO2

    def baseline_emission_setup(self):
        """Make baselines & evaluate them at decision node times.

        TCREZClimate relies on a designated emissions pathway, from which
        we mitigate to limit global warming. In this function, we import every
        SSP baseline from "SSP_baselines.csv".
        (The values in this file were taken from
        https://tntcat.iiasa.ac.at/SspDb/dsd?Action=htmlpage&page=about)

        We then extend them to the year 2400 using the prescription of
        Meinshausen et al., 2020 by basically drawing a straight line from the
        emissions values in 2100 to zero by 2250. (Seriously, look at their
        Figure 2.)

        We then evaluate each of these baselines at the decision period times.
        These will be used throughout TCREZClimate.

        *Raises ValueError if baseline_num is outside acceptable range.
        """

        time, _, run_data = import_csv("SSP_baselines", delimiter=',',
                                                header=True, indices=1)

        # change time from an array of strings to integers
        time = np.array(time, dtype=np.int)

        # check to see if the baseline_num is valid. if it is, select the
        # correct run data from run_data.
        if self.baseline_num > 5 or self.baseline_num <= 0:
            raise ValueError("Invalid baseline_num parameter; must be a value \
                             between 1 and 5.")
        else:
            baseline_gtco2_2100 = run_data[self.baseline_num - 1]

        # make extended baseline in gtco2
        self._make_extension(time, baseline_gtco2_2100,
                             self.tree.decision_times[-1] + 2020)

        # make baseline in ppm CO2
        self.baseline_ppm = self.baseline_gtco2 * self.GTCO2_TO_PPM

        # calculate the cumulative emissions in 1000 GtCO2 of the baseline.
        # Note that the get_integral_var_ub call is multiplied by 10**(-3), as
        # the baseline that is being integrated is in GtCO2, so we multiply the
        # reuslt by 10**(-3) to transform the units.
        self.baseline_cumemit = self.CUMEMIT_2019 + \
                                get_integral_var_ub(self.baseline_gtco2,
                                                    self.times, self.times) * 10**(-3)

        # now evalute the baselines at the node times. node times are years
        # after 2020, our first year. the spacing between emission data points
        # is ten. so dividing dec_times by 10 gives the index of the emission
        # data for that year. evaluating our various baselines at those periods
        # give the emission data at node times.
        self.dec_times_ind = self.tree.decision_times // self.DELTA_T

        # eval baselines at period times
        self.baseline_gtco2_periods = self.baseline_gtco2[self.dec_times_ind]
        self.baseline_ppm_periods = self.baseline_ppm[self.dec_times_ind]
        self.baseline_cumemit_periods = self.baseline_cumemit[self.dec_times_ind]

    def _make_extension(self, time, baseline_gtco2_2100, extension_year):
        """Make baseline extensions from 2100 -> extension_year.

        TCREZClimate requires emissions values through the last decision time.
        The SSP database only provides values through 2100, but in Meinshausen
        et al., 2020 extensions are provided. We make their approximate
        versions here by linearly interpolating the final emission value to
        zero in 2250.

        Parameters
        ----------
        time: nd array
            array of time values for which emissions time series are evaluated

        baseline_gtco2_2100: nd array
            array of emissions values in GtCO2 at times in `time` argument

        extension_year: int
            year for the emissions baseline to be extended to
        """

        # get time difference (= 10 in our case)
        self.DELTA_T = time[1] - time[0]

        # append desired value to time and baselines
        time_appended = np.hstack((time, np.array([2250])))
        baseline_gtco2_appended = np.hstack((baseline_gtco2_2100, np.array([0])))

        # now time is [2010, ..., 2100, 2250] and the emission time series are
        # [val_2010, ..., val_2100, val_2250]. we can now interpolate by making
        # a new set of times to interpolate to.
        time_2100_ext = np.arange(2100 + self.DELTA_T, int(extension_year) +
                                   self.DELTA_T, self.DELTA_T)

        # make new full times that new emissions pathways will be evaluated at
        self.times = np.hstack((time, time_2100_ext))

        # now interpolate and return extended time emissions time series
        self.baseline_gtco2 = np.interp(self.times, time_appended,
                                          baseline_gtco2_appended)

    def get_mitigated_baseline(self, m, node=None, baseline="ppm",
                               is_last=False):
        """Calculate the mitigated version of a given baseline.

        In TCREZClimate, we often need to apply a mitigation to the baseline
        emission pathway. This function creates this "mitigated" baseline by
        multiplying every emissions value between two decision nodes by a
        mitigation value supplied by m. If m is a constant, then we make an
        array of constant mitigations.

        Parameters
        ----------
        m: nd array
            mitigation values; can either be a constant value (with node=None)
            or an array with shape equal to (tree.num_decision_nodes,)

        node: int
            node number. method makes an emission time series *up until and
            including* the time of the node. (None by default.)

        baseline: string
            tells code which baseline to create a mitigated version of, such
            that:
                "ppm": baseline in ppm (default)
                "gtco2": baseline in gtco2
                "cumemit": cumulative emissions baseline (in 1000 gtco2)

        is_last: bool
            is this the final period?

        Returns
        -------
        mitigated_baseline: nd array
            mitigated emissions baseline

        trunc_times: nd array
            times at which the mitigated baseline is evaluated
            * only if node is not None
        """


        # make dictionary of baselines; needed for later
        baseline_dict = {'ppm': self.baseline_ppm, 'gtco2':
                         self.baseline_gtco2, 'cumemit': self.baseline_cumemit}

        # if not None is given as node, then create a truncated, mitigated
        # emissions baseline. 

        # NOTE: the baseline returned here does not include the action at the
        # node in its baseline. this is because the emissions prior to a node
        # should not depend on the action at that node.

        if node is not None:
            # we cannot just take m(t) * C(t) to make a mitigated version of
            # the cumulative emissions baseline because the integral is time
            # dependent. Therefore, we make a mitigated baseline in terms of
            # gtco2 and calculate the integral after we've applied the
            # mitigation
            if baseline == "cumemit":
                tmp_baseline = 'gtco2'
            else:
                tmp_baseline = baseline

            # get period we're in (i.e., how many decisions have we made?)
            period = self.tree.get_period(node)

            if is_last:
                period += 1

            # find path we've taken to get to current node
            path = self.tree.get_path(node)

            # make mitigation for given path
            mit = m[path]

            # index of node time
            node_time_index = self.dec_times_ind[period]

            # make shifted indices to not include action at final node in
            # baseline
            shifted_inds = self.dec_times_ind.copy()
            shifted_inds[1:] += 1

            # truncate times that the mitigated baseline is evaluated at (this
            # can be used for calculating the cumulative emissions after making
            # the 
            trunc_times = self.times[:node_time_index + 1]

            # make empty mitigated baseline
            mitigated_baseline = np.zeros_like(trunc_times, dtype = np.float32)

            # fill in mitigated_baseline
            for i in range(period):
                tmp_ind_low = shifted_inds[i]
                tmp_ind_high = shifted_inds[i+1]
                try:
                    mitigated_baseline[tmp_ind_low:tmp_ind_high] = \
                            baseline_dict[tmp_baseline][tmp_ind_low:tmp_ind_high]\
                            * (1 - mit[i])
                except KeyError:
                    print("Invalid baseline. Only 'ppm', 'gtco2', and 'cumemit'\
                          are implemented.")
                    raise

            if baseline == 'cumemit':
                mitigated_baseline = self.CUMEMIT_2019 + 10**(-3) * \
                                     get_integral_var_ub(mitigated_baseline,
                                                         trunc_times,
                                                         trunc_times)

            return mitigated_baseline, trunc_times

        # otherwise we're after just a scaled up/down version of the baseline,
        # so just make that without the above headache
        else:
            try:
                mitigated_baseline = baseline_dict[baseline] * (1 - m)
                return mitigated_baseline

            except KeyError:
                print("Invalid baseline. Only 'ppm', 'gtco2, and \
                      'cumemit' are implemented.")
                raise
