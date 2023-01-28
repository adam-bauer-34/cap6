"""Damage simulation class.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.14.2022

Damage simulation class for CAP6. Simulates damage pathways for use in
damage.py and in the utility optimization.
"""

import types
try:
    import copyreg
except:
    import copy_reg as copyreg

import numpy as np
import multiprocessing as mp

from src.tools import _pickle_method, _unpickle_method
from src.tools import write_columns_csv, append_to_existing
from .cal.damage_cal import dam_cal_params
from random import randrange

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class DamageSimulation(object):
    """Simulation of damages for the Climate Asset Pricing model -- AR6.

    The damage function simulation is a key input into the pricing engine.
    Damages are represented in arrays of dimension n x p, where n = num states
    and p = num periods. The arrays are created by Monte Carlo simulation. Each
    array specifies for each state and time period a damage coefficient.

    Damages are calculated using a prescribed damage function. The damage
    function maps temperature anomalies above preindustrial (in deg C) to
    economic loss (in percentage of global GDP).

    Parameters
    ----------
    tree : `TreeModel` object
        tree structure used
    emission_baseline: `EmissionBaseline` object
        emission baseline class
    climate: `Climate` object
        provides framework for climate-related calculations
    mitigation_constants: nd array
        set of constants we use for our prototypical atmospheres in the damage
        simulation
    draws : int
        number of samples drawn in Monte Carlo simulation.
    dam_func: int
        tells the code which damage function to use. the choices are:
            - 0: damage function which is concave down and increasing,
            modeled after Burke et al., 2018
            - 1: damage function which is concave up and increasing,
            modeled after Rose et al., 2017
            - 2: damage function which is concave up and increasing, modeled
            after Howard & Sterner, 2017. This one has larger damages than Rose
            et al., 2017
    d_unc: int
        toggles parametric uncertainty in damage functions
    t_unc: int
        toggles parametric uncertainty in temperature emulation
    tip_on: bool
        T/F: tipping point damages included?

    Attributes
    ----------
    tree : `TreeModel` object
        tree structure used
    emission_baseline: `EmissionBaseline` object
        emission baseline class
    climate: `Climate` object
        provides framework for climate-related calculations
    mitigation_constants: nd array
        set of constants we use for our prototypical atmospheres in the damage
        simulation
    draws : int
        number of samples drawn in Monte Carlo simulation.
    dam_func: int
        selects which damage function is used in damage simulation.
            0: simulates all three damage functions with equal probability
            (this method incorporates both model and parameter uncertainty)
            1: concave down damage function, taken from Burke et al., 2015
            2: concave up damage function, taken from Rose et al., 2017
            3: concave up damage function, taken from Howard & Sterner, 2017
    d_unc: int
        toggles parametric uncertainty in damage functions
    t_unc: int
        toggles parametric uncertainty in temperature emulation
    tip_on: bool
        T/F: tipping points damages included?
    d : ndarray
        simulated damages
    """

    def __init__(self, tree, emission_baseline, climate, draws,
                 mitigation_constants, dam_func, tip_on, d_unc, t_unc):
        self.tree = tree
        self.emission_baseline = emission_baseline
        self.climate = climate
        self.draws = draws
        self.mitigation_constants = mitigation_constants
        self.dam_func = dam_func
        self.tip_on = tip_on
        self.d_unc = d_unc
        self.t_unc = t_unc
        self.d = None

    def simulate(self, write_to_file=False,
                 filename="simulated_damages.csv"):
        """Create damage function values in 'p-period' version of the Summers -
        Zeckhauser model.

        Parameters
        ----------
        write_to_file : bool, optional
            wheter to save simulated values
        filename: string
            the name of the simulated damage file being saved

        Returns
        -------
        ndarray
            3D-array of simulated damages

        Note
        ----
        Uses the :mod:`~multiprocessing` package.
        """

        dnum = len(self.mitigation_constants)

        # parallelize using Pool
        pool = mp.Pool(processes=dnum)
        self.d = np.array(pool.map(self._run_path, self.mitigation_constants))

        if write_to_file:
            self._write_to_file(filename)
        return self.d

    def _run_path(self, const_mitigation_val):
        """Run damage simulation for one value of mitigation.

        Using a damage function, we calculate a (num_final_states x
        num_periods) matrix of damage values. Later, this array will be shrunk
        into a tree with the same shape as the TreeModel object.

        Parameters
        ---------
        const_mitigation_val: float
            The constant level of mitigation we're calculating the matrix for.
            (This essentially defines a "prototypical damage pathway" which is
            interpolated over in damage.py.)

        Returns
        -------
        d: ndarray
            num_end_states x num_periods matrix of simulated damage values,
            sorted worst -> best end state damage values.
        """

        # make blank damage values
        d = np.zeros((self.tree.num_final_states, self.tree.num_periods))

        # make temperature pathways
        temp_pathways = self.climate.get_temperature_ts(m0=const_mitigation_val,
                                                        at_periods=True,
                                                        MC=self.t_unc)

        # make bulk damages
        damage_bulk = np.zeros_like(temp_pathways, dtype=np.float32)

        # basic idea here: if dam_func is anything but zero, if statement
        # evaluates to true. so if dam_func != 0, you isolate the damage
        # function simulation. if dam_func is zero, then else is triggered, and
        # all the damage functions are considered.
        if self.dam_func:
            dam_func_dist = np.ones(self.draws) * self.dam_func
        else:
            dam_func_dist = np.array([randrange(1,4) for i in range(0,
                                                                self.draws)],
                                     dtype=np.int)

        # make bool arrays corresponding to each damage function (burke is 1,
        # rose is 2, howard & sterner is 3)
        stats = dam_func_dist == 1
        structs = dam_func_dist == 2
        metas = dam_func_dist == 3

        # feed temperature pathways to damage function for bulk damages and
        # tipping points
        if stats.any():
            damage_bulk[stats, :] = self._damage_function_stat(temp_pathways[stats, :],
                                                                 MC=self.d_unc)

        if structs.any():
            damage_bulk[structs, :] = self._damage_function_struct(temp_pathways[structs, :],
                                                               MC=self.d_unc)

        if metas.any():
            damage_bulk[metas, :] = self._damage_function_meta(temp_pathways[metas, :],
                                                                MC=self.d_unc)

        # if tipping points are on, add their damage to bulk damages
        if self.tip_on:
            damage_tp = self._damage_function_tipping_points(temp_pathways,
                                                             MC=self.d_unc)
            damage = damage_bulk + damage_tp

        else:
            damage = damage_bulk

        # in each of the above, we calculate the damages including our starting
        # point. but for the simulation, we only want the damages past 2020. so
        # cut off the first value.
        damage_past2020 = damage[:, 1:]

        # sort damages based on outcome of simulation
        damage_sorted = self._sort_array(damage_past2020)

        # basic gist of these lines: do a weighted average of the simulated
        # damage matrix. this collapses a matrix of size (draws, num_periods)
        # down to a matrix of (num_final_states, num_periods) 
        weights = self.tree.final_states_prob*(self.draws)
        weights = (weights.cumsum()).astype(int)
        d[0,] = damage_sorted[:weights[0], :].mean(axis=0)
        for n in range(1, self.tree.num_final_states):
            d[n,] = np.maximum(0.0, damage_sorted[weights[n-1]:weights[n],
                                                  :].mean(axis=0))

        # reverse order of matrix so that highest damage paths are first and
        # lowest are at the bottom
        d = d[::-1]
        return d

    def _damage_function_stat(self, temp_pathways, MC):
        """Statistically estimated  damage function.

        This damage function is created by fitting Figure Cross-Working
        Group Box ECONOMIC.1 data in the IPCC Report Ch. 2, page 16-114. The
        result is a function of economic damages (in percentage of GDP) as
        a function of temperature anomaly above preindustrial. This damage
        function is concave down.

        To see how the coefficients below are calculated, see my notes. (Ask
        Adam.)

        Parameters
        ----------
        temp_pathways: nd array
            matrix of temperature pathways for different values of TCRE
        MC: int
            toggles parametric uncertainty in damage function

        Returns
        -------
        damage: nd array
            damage pathways after plugging temperature time series into damage
            function
        """

        number_of_draws, _ = np.shape(temp_pathways)

        # import parameters 
        if self.emission_baseline.baseline_num == 1:
            D1_mid_mean, D1_mid_std, conc_mag_mid, D1_end_mean, D1_end_std,\
                conc_mag_end = dam_cal_params["burke_ssp1"]

        elif self.emission_baseline.baseline_num == 2:
            D1_mid_mean, D1_mid_std, conc_mag_mid, D1_end_mean, D1_end_std,\
                conc_mag_end = dam_cal_params["burke_ssp2"]

        elif self.emission_baseline.baseline_num == 3:
            D1_mid_mean, D1_mid_std, conc_mag_mid, D1_end_mean, D1_end_std,\
                conc_mag_end = dam_cal_params["burke_ssp3"]

        elif self.emission_baseline.baseline_num == 4:
            D1_mid_mean, D1_mid_std, conc_mag_mid, D1_end_mean, D1_end_std,\
                conc_mag_end = dam_cal_params["burke_ssp4"]

        elif self.emission_baseline.baseline_num == 5:
            D1_mid_mean, D1_mid_std, conc_mag_mid, D1_end_mean, D1_end_std,\
                conc_mag_end = dam_cal_params["burke_ssp5"]

        b2_mid, b1_mid = self._get_fitted_param_dists(D1_mid_mean, D1_mid_std,
                                                      conc_mag_mid,
                                                      number_of_draws)

        b2_end, b1_end = self._get_fitted_param_dists(D1_end_mean, D1_end_std,
                                                      conc_mag_end,
                                                      number_of_draws)

        # if no MC, take means of resulting coefficients
        if not MC:
            # mid century
            b2_mid = np.array([np.mean(b2_mid) for i in range(number_of_draws)])
            b1_mid = np.array([np.mean(b1_mid) for i in range(number_of_draws)])

            # end of century
            b2_end = np.array([np.mean(b2_end) for i in range(number_of_draws)])
            b1_end = np.array([np.mean(b1_end) for i in range(number_of_draws)])

        damage = np.zeros_like(temp_pathways, dtype=float)

        damage[:, :3] = temp_pathways[:, :3] * (b2_mid[:, None] * temp_pathways[:, :3]\
                                          + b1_mid[:, None])

        damage[:, 3:] = temp_pathways[:, 3:] * (b2_end[:, None] * temp_pathways[:, 3:]\
                                          + b1_end[:, None])

        damage[damage < 0] = 0.0
        return damage

    def _damage_function_struct(self, temp_pathways, MC):
        """Structurally estimated damage function

        This damage function is created in the same way as
        _damage_function_burke, except now we use Rose et al.'s  curve,
        which is concave up.

        Again, ask Adam to see how the coefficients are calculated.

        Parameters
        ----------
        temp_pathways: nd array
            matrix of temperature pathways for different values of TCRE
        MC: int
            toggles parametric uncertainty in damage function

        Returns
        -------
        damage: nd array
            damage time series after plugging in temperature time series into
            damage function
        """

        number_of_draws, _ = np.shape(temp_pathways)

        D1_mean, D1_std, conc_mag = dam_cal_params["structural"]

        b2, b1 = self._get_fitted_param_dists(D1_mean, D1_std, conc_mag,
                                              number_of_draws)

        # if no MC, take means of resulting coefficients
        if not MC:
            # mid century
            b2 = np.array([np.mean(b2) for i in range(number_of_draws)])
            b1 = np.array([np.mean(b1) for i in range(number_of_draws)])

        damage = temp_pathways * (b2[:, None] * temp_pathways + b1[:, None])
        damage[damage < 0] = 0.0
        return damage

    def _damage_function_meta(self, temp_pathways, MC):
        """Meta analytic damage function.

        This damage function is fit in the same way as the Burke and Rose
        damage functions. The data is taken from the Appendix of the original
        Howard & Sterner paper, unlike Burke & Rose, where the data was taken
        from the IPCC plot.

        Parameters
        ----------
        temp_pathways: nd array
            matrix of temperature pathways with different values of TCRE
        MC: int
            toggles parametric uncertainty in damage function

        Returns
        -------
        damage: nd array
            damage time series after applying damage function
        """

        number_of_draws, _ = np.shape(temp_pathways)

        D1_mean, D1_std, conc_mag = dam_cal_params["meta-analytic"]

        b2, b1 = self._get_fitted_param_dists(D1_mean, D1_std, conc_mag,
                                              number_of_draws)

        # if no MC, take means of resulting coefficients
        if not MC:
            # mid century
            b2 = np.array([np.mean(b2) for i in range(number_of_draws)])
            b1 = np.array([np.mean(b1) for i in range(number_of_draws)])

        damage = temp_pathways * (b2[:, None] * temp_pathways + b1[:, None])
        damage[damage < 0] = 0.0
        return damage

    def _damage_function_tipping_points(self, temp_pathways, MC):
        """Add-on damage function for tipping points.

        This method contains an additional damage function found in Dietz et
        al., 2021. doi: https://doi.org/10.1073/pnas.2103081118

        We take the polynomial from Fig. 5c and use it here, after converting
        it from % initial consumption to % global GDP.

        NOTE: the coefficients given in Dietz et al are in units of percentage
        of consumption, not fraction. We use fractional units in our work, thus
        the additional factor of 0.01 in L351 (i.e., in damage_per_glob_gdp
        equation).

        Parameters
        ----------
        temp_pathways: nd array
            matrix of temperature pathways with different values of TCRE
        MC: int
            toggles parametric uncertainty in damage function

        Returns
        -------
        damage: nd array
            damage time series after applying damage function
        """

        number_of_draws, _ = np.shape(temp_pathways)

        CONSUMP_INIT = 61.88 # trillion, from World Bank
        GDP_INIT = 84.75 # trillion, from World Bank

        # if MC is on, we distribute D_1 vals. if not, make an array of
        # constant vals.
        if MC:
            b_dist = np.random.normal(loc=0.48, scale=0.02,
                                  size=number_of_draws)
            a_dist = np.random.normal(loc=-0.04, scale=0.01,
                                  size=number_of_draws)
        else:
            b_dist = np.array([0.48 for i in range(number_of_draws)])
            a_dist = np.array([-0.04 for i in range(number_of_draws)])

        damage_per_consump = temp_pathways * (b_dist[:, None] * temp_pathways\
                                              + a_dist[:, None])
        damage_per_gdp = damage_per_consump * GDP_INIT**(-1) * CONSUMP_INIT\
                       * 0.01

        damage_per_gdp[damage_per_gdp < 0] = 0.0
        return damage_per_gdp

    def _get_fitted_param_dists(self, D_1d_mean, D_1d_std, conc_mag, size):
        """Get fitted damage function parameters.

        This function generates the distribution of coefficients for the
        quadratic fitting function to each damage function. For detais on what
        all the below means, ask Adam.

        Parameters
        ----------
        D_1d_mean: float
            mean of D_1d distribution

        D_1d_std: float
            standard deviation of D_1d distribution

        conc_mag: float
            magnitude of the concavity; < 1 implies concave down, > 1 concave
            up, = 1 is linear. the further the magnitude is away from 1 in
            either, the "more" concave up/down the resulting curve is.

        size: int
            number of draws to generate

        Returns
        -------
        b2: (size,) numpy array
            distribution of second order coefficient

        b1: (size,) numpy array
            distribution of first order coefficient
        """

        T_1 = 3
        T_2 = 10

        D_1d = np.random.normal(loc=D_1d_mean, scale=D_1d_std, size=size)
        D_2d = D_1d * (T_2 * T_1**(-1)) * conc_mag

        denom = T_1 * T_2 * (T_2 - T_1)
        b2 = (D_2d * T_1 - D_1d * T_2) * denom**(-1)
        b1 = (D_1d * T_2**2 - D_2d * T_1**2) * denom**(-1)

        return b2, b1


    def _sort_array(self, array):
        """Sort array of size (x, y) by the value of each row's final argument.
        """

        return array[array[:, self.tree.num_periods-1].argsort()]

    def _write_to_file(self, filename):
        """Write results of simulation to file.

        Parameters
        ----------
        filename: string
            the name of the file to be saved
        """

        write_columns_csv(self.d[0].T, filename)
        for arr in self.d[1:]:
            append_to_existing(arr.T, filename, start_char='#')
