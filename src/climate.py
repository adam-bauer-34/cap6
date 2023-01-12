"""Climate class for TCREZClimate.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.21.2022

This code contains the climate class for TCREZClimate. It utilizes an impulse
response function to calculate the concentrations at a given time for a given
emission pathway. It also samples the distribution of TCRE.
"""

import numpy as np

from abc import ABCMeta, abstractmethod
from scipy.stats import truncnorm

class Climate(object, metaclass=ABCMeta):
    """Abstract climate class for the TCREZClimate model.

    Parameters
    ----------
    tree: `TreeModel` object
        provides tree structure of the model
    emit_baseline: `EmissionBaseline` object
        baseline of emissions object

    Attributes
    ----------
    tree: `TreeModel` object
        provides tree structure of model
    emit_baseline: `EmissionBaseline` object
        provides emission baseline of model
    """

    def __init__(self, tree, emit_baseline):
        self.tree = tree
        self.emit_baseline = emit_baseline

    @abstractmethod
    def get_conc_and_forcing_at_node(self):
        """Get CO2 concentrations and forcing at a given node.
        """
        pass

    @abstractmethod
    def get_temperature_ts(self):
        """Get temperature as a function of time.
        """
        pass

class BPWClimate(Climate):
    """Climate model for TCREZClimate.

    This contains multiple methods and attributes which describe the climate
    system as of AR6.

    Parameters
    ----------
    tree: `TreeModel` object
        tree structure of the model

    emit_baseline: `EmissionBaseline` object
        emission baseline

    draws: int
        Number of Monte Carlo samples to take of relevant distributions.

    t_var_mult: float
        multiple of TCRE standard deviation (used primarily for probing impacts
        of larger climate uncertainty in the damage simulation) (default is 1.)

    Attributes
    ----------
    The following climate parameters:
        A_0: float
            Zeroth order coefficient of the Joos et al impulse response
            function (IRF)
        A_1: float
            First order coefficient of the Joos et al IRF
        A_2: float
            Second order coefficient of the Joos et al IRF
        A_3: float
            Third order coefficient of the Joos et al IRF
        TAU_1, TAU_2, TAU_3: floats
            timescales for the Joos et al IRF (in years)
        C_0: float
            preindustrial CO2 concentrations (in ppm)
        C_2020: float
            CO2 concentrations in 2020 (in ppm)
        F_0: float
            forcing equation coefficient; F(t) = F_0 ln(C(t)/C_0)
            (in W m^-2)
        TCRE_BEST_ESTIMATE: float
            best estimate of the transient climate response to emissions (in K
            / 1000 GtCO2)
            (Taken from AR6 WG1 SPM-36.)
        TCRE_STD: float
            standard deviation of the TCRE distribution (in K / 1000 GtCO2)
            (Taken from AR6 WG1 SPM-36.)
    t_var_mult: float
        multiple of TCRE standard deviation (used primarily for probing
        impacts of larger climate uncertainty in the damage simulation)
        (default is 1.)

    Properties
    ----------
    TCRE_dist: (self.draws,) array
        Distribution of TCRE values (in K / 1000 GtCO2), uses
        scipy.stats.truncnorm to generate a truncated normal distribution (gets
        rid of non-physical negative values of TCRE).

    Methods
    -------
    get_conc_and_forcing_at_node:
        returns concentrations and/or forcing at a given node for a mitigation
        vector
    get_conc_at_node/get_forcing_at_node:
        returns conc/forcing at node for a mitigation vector (calls
        get_conc_and_forcing_at_node with returning argument defaulted to
        either "conc" or "forcing")
    get_temperature_ts:
        calculates the temperature time series for a given method
    _get_temperature_ts_TCRE:
        calculate temperature time series using TCRE formalism
    _make_joos_IRF:
        makes impulse response function from Joos et al., 2013
    """

    def __init__(self, tree, emit_baseline, draws, t_var_mult=1.0):
        self.tree = tree
        self.emit_baseline = emit_baseline
        self.DRAWS = draws
        self.t_var_mult = t_var_mult

        # forcing equation constants
        self.C_0 = 278 # ppm
        self.F_0 = 5.35 # W m^-2
        self.C_2020 = 420.87 # ppm (from co2.earth/daily-co2)

        # Coefficients of exponentials in fit done by Joos et al., 2013
        self.A_0 = 0.2173
        self.A_1 = 0.2240
        self.A_2 = 0.2824
        self.A_3 = 0.2763

        # 3 timescales calculated in Joos et al., 2013
        self.TAU_1 = 394.4 # years
        self.TAU_2 = 36.54 # years
        self.TAU_3 = 4.304 # years

        # TCRE best estimates from AR6 WG1 SPM-36
        self.TCRE_BEST_ESTIMATE = 0.45 # deg C / 1000 GtCO2
        self.TCRE_STD = 0.18 * self.t_var_mult # deg C / 1000 GtCO2

    @property
    def TCRE_dist(self):
        """Sample the TCRE distribution. (Assumed to be a truncated Gaussian.)
        """

        a = -1 * self.TCRE_BEST_ESTIMATE * self.TCRE_STD**(-1)
        b = (1000 - self.TCRE_BEST_ESTIMATE) * self.TCRE_STD**(-1)

        tcre_dist = truncnorm.rvs(a, b, loc=self.TCRE_BEST_ESTIMATE,
                                     scale=self.TCRE_STD, size=self.DRAWS)

        return tcre_dist

    def get_temperature_ts(self, m0=0, at_periods=False, MC=True, method=0):
        """Get temperature time series.

        Generate temperature time series. If MC=True then we return self.draws
        number of time series. Otherwise, we only return one time series.

        Parameters
        ----------
        m0: float
            constant value of mitigation to scale the emission baseline by.
            (default is 0.)
        at_periods: bool
            whether or not we evaluate the temperature time series at period
            times
        MC: bool
            toggles Monte Carlo simulation. if False, mean values of climate
            parameters are used to compute the tempearture time series.
            (default is True.)
        method: int
            toggles method used to calculate the temperature time series, such
            that:
                0: TCRE
                1: Geoffroy et al. 2 box method (future implementation)
            (default is 0, TCRE.)

        Returns
        -------
        temperature_ts: nd array
            temperature time series
        """

        # TCRE method
        if method == 0:
            temperature_ts = self._get_temperature_ts_TCRE(m0, MC, at_periods)

        # Geoffroy method (for future implementation)
        elif method == 1:
            temperature_ts = None

        return temperature_ts

    def _get_temperature_ts_TCRE(self, m0, MC, at_periods):
        """Get temperature time series using TCRE.

        Generates temperature time series using TCRE, such that:
            T(t) = \lambda * (1 - m0) * cumulative_emissions(t)

        Parameters
        ----------
        m0: float
            constant mitigation value
        MC: bool
            toggles Monte Carlo simulation. if False, mean values of climate
            parameters are used to compute the temperature time series.
        at_periods: bool
            true if we evalute the temperature time series at period times

        Returns
        -------
        temperature_ts: nd array
            temperature time series
        """

        # note to self: these temperature pathways are in terms of temperature
        # change *above preindustrial*! 

        if MC:
            temperature_ts = (self.emit_baseline.baseline_cumemit\
                           - self.emit_baseline.baseline_cumemit[0])\
                           * (1 - m0) * self.TCRE_dist[:, None]
            if at_periods:
                temperature_ts = temperature_ts[:,
                                                self.emit_baseline.dec_times_ind]

        else:
            # if no MC, make an array of the same TCRE value
            TCRE_vals = np.ones_like(self.TCRE_dist, dtype=np.float32) *\
                        self.TCRE_BEST_ESTIMATE
            temperature_ts = (self.emit_baseline.baseline_cumemit\
                              - self.emit_baseline.baseline_cumemit[0])\
                              * (1 - m0) \
                              * TCRE_vals[:, None]
            if at_periods:
                temperature_ts = temperature_ts[:,
                                                self.emit_baseline.dec_times_ind]

        return temperature_ts

    def get_conc_and_forcing_at_node(self, m, node, returning="both",
                                     is_last=False):
        """Calculate the carbon concentrations and radiative forcing at a node.

        Using the impulse response function from Joos et al., 2013 and the
        common equation F(t) = F0 ln(C(t)/C0) to calculate the forcing.

        Parameters
        ----------
        m: nd array
            mitigation vector up to and including node

        node: int
            node number

        returning: string
            tells function what to return, such that:
                "conc": returns concentrations only.
                "forcing": returns forcing only.
                "both": returns both as a tuple (conc, forcing). (DEFAULT.)

        is_last: bool
            is it the last period?

        Returns
        -------
        conc_node: float
            carbon concentrations at node

        forcing_node: float
            forcing at node
        """

        if node == 0:
            if returning == "conc":
                return self.C_2020
            elif returning == "forcing":
                return 0.0
            elif returning == "both":
                return self.C_2020, 0.0

        # get mitigated baseline
        mitigated_baseline, trunc_times = \
            self.emit_baseline.get_mitigated_baseline(m=m, node=node,
                                                      baseline="ppm",
                                                      is_last=is_last)
        # time setup 
        # we want an equal points on either side of zero to properly do convolution
        FINAL_TIME = trunc_times[-1] - 2020
        time = np.arange(-FINAL_TIME, FINAL_TIME, 1)

        # parse time range
        time_after_2020 = time[FINAL_TIME:]
        time_before_2020 = time[:FINAL_TIME]

        # now interpolate bau pathway from t = 0 to t = FINAL_TIME
        interp_bau = np.interp(time_after_2020, trunc_times - 2020,
                               mitigated_baseline)

        # make emission pathway zero before 2015 and emission_path after 2015
        # the "half zeros" are required to get a consistent convolution
        # with the IRF below
        bau_interp_full_path = np.hstack((np.zeros_like(time_before_2020), interp_bau))

        # make Joos et al 2013 IRF with zeros before 2015 as was done for emission path
        joos_IRF = self._make_joos_IRF(time_after_2020, time_before_2020)

        # convolve IRF and interpolated BAU to make concentrations time series
        # NOTE: bau_interp_full_path is shifted by one to make the convolution 
        # start at t = 0. Idea taken from:
        # https://stackoverflow.com/questions/63657882/why-does-np-convolve-shift-the-resulted-signal-by-1
        convolve_bau_IRF = np.convolve(bau_interp_full_path[1:], joos_IRF, mode='same')

        # make preindustrial component of concentrations pathway
        preindustrial_post_2020 = self.C_2020 * (self.A_0 + self.A_1)**(-1) * \
            (self.A_0 + self.A_1 * np.exp(-time_after_2020 / self.TAU_1))
        total_preindustrial = np.hstack((np.zeros(FINAL_TIME),
                                         preindustrial_post_2020))

        # calculate the total concentration pathway
        total_conc = total_preindustrial + convolve_bau_IRF

        # final value in total_conc is the concentration at the node
        conc_node = total_conc[-1]

        if conc_node > self.C_0:
            forcing_node = self.F_0 * np.log(conc_node/self.C_0)
        else:
            # second order taylor expansion of forcing equation when conc_node < C0
            forcing_node = self.F_0 * (conc_node - self.C_0) \
                    * self.C_0**(-1) - self.F_0 * (conc_node - self.C_0)**2 \
                    * (2 * self.C_0**2)**(-1)

        # return values based on value of returning parameter
        if returning == "both":
            return conc_node, forcing_node

        elif returning == "forcing":
            return forcing_node

        elif returning == "conc":
            return conc_node

    def _make_joos_IRF(self, time_after_2015, time_before_2015):
        """Make Joos et al., 2013 Impulse response function

        Makes Joos et al IRF.

        Parameters
        ----------
        time_after_2015: nd array
            time values after 2015 to evaluate the IRF on
        time_before_2015: nd array
            times to set to zero

        Returns
        -------
        joos IRF: nd array
            the IRF according to Joos et al., 2013
        """

        # make IRF terms
        joos_IRF_0th = self.A_0
        joos_IRF_1st = self.A_1 * np.exp(-time_after_2015 / self.TAU_1)
        joos_IRF_2nd = self.A_2 * np.exp(-time_after_2015 / self.TAU_2)
        joos_IRF_3rd = self.A_3 * np.exp(-time_after_2015 / self.TAU_3)

        # add each term to get final IRF
        joos_IRF_after_2015 = joos_IRF_0th + joos_IRF_1st + joos_IRF_2nd + joos_IRF_3rd
        joos_IRF = np.hstack((np.zeros_like(time_before_2015), joos_IRF_after_2015))
        return joos_IRF

    def get_conc_at_node(self, m, node, returning="conc", is_last=False):
        """Get CO2 concentrations at a node.

        Calls get_conc_and_forcing_at_node with returning set to "conc". See
        get_conc_and_forcing_at_node for argument descriptions.

        Returns
        -------
        conc_node: float
            CO2 concentrations at the node.
        """

        conc_node = self.get_conc_and_forcing_at_node(m, node, returning,
                                                      is_last)
        return conc_node

    def get_forcing_at_node(self, m, node, returning="forcing", is_last=False):
        """Get forcing at node.

        Calls get_conc_and_forcing_at_node with returning set to "forcing". See
        get_conc_and_forcing_at_node for argument descriptions.

        Returns
        -------
        forcing_node: float
            radiative forcing at the node.
        """

        forcing_node = self.get_conc_and_forcing_at_node(m, node, returning,
                                                         is_last)
        return forcing_node
