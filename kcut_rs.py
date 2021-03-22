##############################################################################
# FILE:    kcut_rs.py
# AUTHOR:  A.C. Deshpande
# PURPOSE: Code to test k-cut required to preserve the validity of reduced
#          shear for Stage IV experiments. Can also calculate Doppler-shift
#          correction for cosmic shear angular power spectra.
##############################################################################

# ============================================================================
# Imports
# ============================================================================

import numpy as np
import camb
import copy
import subprocess
import pickle

from scipy import interpolate
from scipy import integrate
from scipy import misc
from BNT import BNT
from astropy import units
from astropy import cosmology
from multiprocessing import Pool

# ============================================================================
# Utility functions
# ============================================================================

class ShearObservable:
    """
    Class for cosmic shear observables for a Euclid-like survey.
    """

    def __init__(self, omega_cdm=0.27, omega_b=0.05, h=0.67,
                 processors=100, spectral_ind=0.96, w0=-1.0, wa=0.0,
                 As=2.115e-9, A_IA=1.72, C_IA=0.0134,
                 def_ells=np.logspace(1.0, 3.7, 25), survey='photometric',
                 recompute_bispec_base=False, verbose=True):
        """
        Parameters
        ----------
        omega_cdm: float
           Dimensionless cold dark matter density.
        omega_b: float
            Dimensionless baryonic matter density.
        h: float
            Hubble parameter
        spectral_ind: float
            Spectral index of primordial power spectrum.
        w0: float
            Present-day dark energy equation-of-state parameter.
        wa: float
            Early Universe dark energy equation-of-state parameter.
        As: float
            Amplitude of power spectrum at pivot scale.
        A_IA: float
            NLA intrinsic alignment model A parameter.
        C_IA: float
            NLA intrinsic alignment model C parameter.
        def_ells: array
            \ell-modes at which to explicitly compute all quantities at.
        survey: str
            Type of WL survey - 'photometric' for standard photometric survey
            or 'kinematic' for theoretical Tully-Fisher survey.
        recompute_bispec_base: bool
            Option to recompute matter bispectrum from scratch with Bihalofit.
        verbose: bool
            Option to print verbose output.
        """

        self.om_cdm = omega_cdm
        self.om_b = omega_b
        self.om_m = omega_b + omega_cdm
        self.om_de = 1.0 - self.om_m
        self.h = h
        self.H0 = h * 100.0
        self.proc = processors
        self.ns = spectral_ind
        self.w0 = w0
        self.wa = wa
        self.As = As
        self.A_IA = A_IA
        self.C_IA = C_IA
        self.t_ells = def_ells
        self.comp_bi = recompute_bispec_base
        self.verbose = verbose
        self.survey = survey

        self.c_light = 3.0e5

        if self.survey not in ['photometric', 'kinematic']:
            raise Exception('Survey must be either photometric or kinematic.')

        if self.survey == 'photometric':
            self.n_gal = 30.0
            bin_limits_list = [0.001, 0.418, 0.560, 0.678, 0.789, 0.900,
                               1.019, 1.155, 1.324, 1.576, 2.50]
        elif self.survey == 'kinematic':
            self.n_gal = 1.1
            bin_limits_list = [0.001, 0.568, 0.654, 0.723, 0.788, 0.851,
                               0.921, 0.999, 1.097, 1.243, 1.68]

        self.survey_min = 0.001
        self.survey_max = 4.0
        self.n_pts = 100
        self.zs = np.linspace(self.survey_min, self.survey_max, self.n_pts)

        chis_fsh = []
        for rst in np.arange(0.001, 4.0, 0.1):
            chis_fsh.append(self.z_to_chi(rst))
        np.save('./temp_io/fsh_comov_vals.npy', chis_fsh)
        if self.verbose is True:
            print('# Saved comoving distance interpolation values for kcut '
                  'calculations.')

        z_load = np.loadtxt('z.txt')
        chi_load = []
        for rshift in z_load:
            chi_load.append(self.z_to_chi(rshift))
        chi_load = np.array(chi_load)
        chi_obj = np.interp(self.zs, z_load, chi_load)

        self.n_funcs = []
        self.n_BNT_list = []
        if self.survey == 'photometric':
            for bin_ind in range(10):
                cur_dist = self.prob_convolve(bin_limits_list[bin_ind],
                                              bin_limits_list[bin_ind+1])
                n_i_vals = cur_dist(self.zs)
                self.n_funcs.append(cur_dist)
                self.n_BNT_list.append(n_i_vals)
        elif self.survey == 'kinematic':
            for bin_ind in range(10):
                cur_dist = self.n_kinematic(bin_limits_list[bin_ind],
                                     bin_limits_list[bin_ind+1])
                n_i_vals = cur_dist(self.zs)
                self.n_funcs.append(cur_dist)
                self.n_BNT_list.append(n_i_vals)

        if self.verbose is True:
            print('# Computed n(z).')
        B = BNT(self.zs, chi_obj, self.n_BNT_list)
        BNT_matrix = B.get_matrix()
        self.BNT_mat = BNT_matrix
        if self.verbose is True:
            print('# Computed BNT matrix.')

        self.skdict = None
        self.power = None
        self.power_linear = None
        self.D_growth = None
        self.Hubble_func = None
        self.E_func = None
        self.power_generator()
        if self.verbose is True:
            print('# Computed power spectrum and related quantities.')

        self.kern_list = []
        for index in range(10):
            inps = []
            for pind in range(self.n_pts):
                inps.append([self.zs[pind], index+1])
            p = Pool(self.proc)
            kern_i = p.starmap(self.kernel_gamma, inps)
            p.close()
            p.join()

            self.kern_list.append(kern_i)
        if self.verbose is True:
            print('# Computed lensing kernels.')

        kern_array = np.array(self.kern_list)
        self.kern_i_BNT_array = np.zeros_like(kern_array)
        ns_array = np.array(self.n_BNT_list)
        self.n_i_BNT_array = np.zeros_like(ns_array)
        for j in range(self.n_pts):
            self.kern_i_BNT_array[:, j] = np.dot(BNT_matrix, kern_array[:, j])
            self.n_i_BNT_array[:, j] = np.dot(BNT_matrix, ns_array[:, j])
        if self.verbose is True:
            print('# Computed BNT transformed kernels.')

        if self.comp_bi is True:
            self.bl_base_gen()
        self.b_mat_ld = np.load('./temp_io/bispec_base.npy')
        if self.verbose is True:
            print('# Loaded matter bispectrum matrix.')

        self.kern_BNT_interp_list = []
        for kn in range(len(self.kern_i_BNT_array)):
            interp_obj = interpolate.InterpolatedUnivariateSpline(self.zs,
                         self.kern_i_BNT_array[kn,:])
            self.kern_BNT_interp_list.append(interp_obj)
        if self.verbose is True:
            print('# Interpolated BNT kernels for RS calculation.')

        return

    def z_to_chi(self, z):
        """
        Function to go from redshift to comoving distance for flat wCDM
        cosmology.

        Parameters
        ----------
        z: float
            Redshift to convert

        Returns
        -------
        float
            Comoving distance at input redshift.
        """
        if z == 0.0:
            chi = 0.0
        else:
            cosmo = cosmology.w0waCDM(H0=100*self.h, Om0=self.om_m,
                                      Tcmb0=2.726,
                                      Ob0=self.om_b, Neff=3.04,
                                   w0=self.w0, wa=self.wa, Ode0=self.om_de)
            chi = cosmo.comoving_distance(z).value
        return chi

    def chi_to_z(self, chi, zmin=0.0):
        """
        Function to go from comoving distance to redshift for flat wCDM
        cosmology.

        Parameters
        ----------
        chi: float
            Comoving distance to convert.
        zmin: float
            Minimum redshift bound for conversion, in case of degenerate
            possible redshifts.

        Returns
        -------
        float
            Redshift at input comoving distance.
        """
        cosmo = cosmology.w0waCDM(H0=100 * self.h, Om0=self.om_m,
                                  Tcmb0=2.726, Ob0=self.om_b, Neff=3.04,
                                  w0=self.w0, wa=self.wa, Ode0=self.om_de)
        z_check = cosmology.z_at_value(cosmo.comoving_distance, chi *
                                       units.Mpc, zmin=zmin, zmax=15.0)
        return z_check

    def n_kinematic(self, bin_z_min, bin_z_max, int_step=10):
        """
        Implements the normalised predicted source distribution of a
        kinematic WL survey, according to Eq. 23
        of https://arxiv.org/abs/1311.1489.

        Parameters
        ----------
        bin_z_min: float
            Lower limit of redshift bin.
        bin_z_max: float
            Upper limit of redshift bin.
        int_step: int
                  Number of steps for integration over redshift.

        Returns
        -------
        object
           Normalised n(z) for kinematic source distribution.
        """
        z_list = np.linspace(bin_z_min, bin_z_max, int_step)

        norm_int = integrate.quad(self.n_kinematic_int, a=bin_z_min,
                                  b=bin_z_max)
        n_nums_list = []
        for zind in range(len(z_list)):
            fin_n = self.n_kinematic_int(z_list[zind]) / norm_int[0]
            n_nums_list.append(fin_n)
        bin_obj = interpolate.InterpolatedUnivariateSpline(z_list,
                                                           np.array(
                                                               n_nums_list),
                                                           ext=1)
        return bin_obj

    def n_kinematic_int(self, z):
        """
        Implements the unnormalised predicted source distribution of a
        kinematic WL survey, according to Eq. 23
        of https://arxiv.org/abs/1311.1489.

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.

        Returns
        -------
        float
           Unnormalised n(z) for kinematic source distribution.
        """

        alpha = 29.98146120631584
        z0 = 1.1025687918762867e-06
        beta = 0.3337510838243506

        pz = (z**alpha) * np.exp(-(z/z0)**beta)
        return pz

    def n_euclidlike(self, z, n_gal=30.0):
        """
        Implements true, normalised galaxy source distribution for a
        Euclid-like survey according to Eq. 113 of
        https://arxiv.org/abs/1910.09273.

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.
        n_gal: float
           Galaxy surface density of survey.
           Euclid-like default - 30 arcmin^{-2}.

        Returns
        -------
        float
           n(z) for Euclid-like source distribution.
        """
        n_dist_int = integrate.quad(self.n_euclidlike_int, a=self.survey_min,
                                    b=self.survey_max)
        prop_con = n_gal / n_dist_int[0]
        fin_n = prop_con * self.n_euclidlike_int(z=z)
        return fin_n

    def n_euclidlike_int(self, z, zm=0.9):
        """
        Integrand for true galaxy source distribution as for a Euclid-like
        survey. See Eq. 113 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.
        zm: float
           Median redshift of survey. Euclid-like default = 0.9.

        Returns
        -------
        float
           Unnormalised n(z) for Euclid-like source distribution.
        """
        z0 = zm / np.sqrt(2.0)
        n_val = ((z / z0)**2.0) * np.exp((-(z / z0)**1.5))
        return n_val

    def prob_photometric(self, z_photo, z, c_b=1.0, z_b=0.0, sigma_b=0.05,
                         c_outlier=1.0, z_outlier=0.1, sigma_outlier=0.05,
                         outlier_frac=0.1):
        """
        Probability distribution function, describing the probability that
        a galaxy with redshift z has a measured redshift z_photo.

        Parameters
        ----------
        z_photo: float
            Measured photometric redshift
        z:  float
            True redshift
        c_b: float
            Multiplicative bias on sample with well-measured redshift.
            Euclid-like default = 1.0.
        z_b: float
            Additive bias on sample with well-measured redshift.
            Euclid-like default = 0.0.
        sigma_b: float
            Sigma for sample with well-measured redshift.
            Euclid-like default = 0.05.
        c_outlier: float
            Multiplicative bias on sample of catastrophic outliers.
            Euclid-like default = 1.0.
        z_outlier: float
            Additive bias on sample of catastrophic outliers.
            Euclid-like default = 0.1.
        sigma_outlier: float
            Sigma for sample of catastrophic outliers.
            Euclid-like default = 0.05.
        outlier_frac: float
            Fraction of catastrophic outliers.
            Euclid-like default = 0.1.

        Returns
        -------
        float
            Probability density.
        """

        fac1 = ((1.0 - outlier_frac) / (np.sqrt(2.0 * np.pi) *
                                        sigma_b * (1.0 + z)))
        fac2 = (outlier_frac / (np.sqrt(2.0 * np.pi) *
                                sigma_outlier * (1.0 + z)))

        p_val = ((fac1 * np.exp((-0.5) * ((z - (c_b * z_photo) - z_b) /
                                          (sigma_b * (1.0 + z)))**2.0)) +
                 (fac2 * np.exp((-0.5) * ((z - (c_outlier * z_photo) -
                                           z_outlier) /
                                          (sigma_outlier * (1.0 + z)))**2.0)))

        return p_val

    def nz_photmetric_integral(self, z, bin_z_max, bin_z_min):
        """
        Defines integrand to calculate denominator of photometric uncertainty
        convolution.

        Parameters
        ----------
        z: float
           Redshift
        bin_z_max: float
                   Upper limit of bin
        bin_z_min: float
                   Lower limit of bin

        Returns
        -------
        float
            Denominator integrand value.
        """
        ret_val = self.n_euclidlike(z) * integrate.quad(self.prob_photometric,
                                                        a=bin_z_min,
                                                        b=bin_z_max,
                                                        args=(z))[0]
        return ret_val

    def prob_convolve(self, bin_z_min, bin_z_max, int_step=0.1):
        """
        Computes the observed galaxy distribution, by carrying out convolution
        of true n(z) with photometric uncertainty PDF.

        Parameters
        ----------
        bin_z_max: float
                   High z limit of bin.
        bin_z_min: float
                   Low z limit of bin.
        int_step: float
                  Redshift step size for interpolating final n(z).
                  Default = 0.1.

        Returns
        -------
        object
            Observed n(z) with photometric redshift uncertainties accounted.
        """
        z_list = np.arange(self.survey_min, self.survey_max, int_step)
        n_nums_list = []
        for zind in range(len(z_list)):
            finprod = (self.n_euclidlike(z_list[zind]) *
                       integrate.quad(self.prob_photometric, a=bin_z_min,
                                      b=bin_z_max, args=(z_list[zind]))[0])
            n_nums_list.append(finprod)

        n_denom = integrate.quad(self.nz_photmetric_integral, a=z_list[0],
                                 b=z_list[-1], args=(bin_z_max, bin_z_min))[0]

        res = np.array(n_nums_list) / n_denom

        obs_bin = interpolate.InterpolatedUnivariateSpline(z_list, res, ext=1)

        return obs_bin

    def power_generator(self):
        """
        Computes matter power spectrum, growth factor, and Hubble function,
        by interfacing with CAMB.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        cp = camb.set_params(ns=abs(self.ns), H0=self.h * 100.0,
                             ombh2=self.om_b * (self.h ** 2),
                             omch2=self.om_cdm * (self.h ** 2), w=-1.0,
                             omk=0.0,
                             wa=0.0, lmax=5000, WantTransfer=True,
                             dark_energy_model='DarkEnergyPPF', mnu=0.06,
                             As=self.As, TCMB=2.726, YHe=0.25, kmax=100.0,
                             halofit_version='mead',
                             redshifts=np.linspace(0.0, 5.0, 50).tolist())
        results = camb.get_results(cp)

        PKcamb = results.get_matter_power_interpolator(nonlinear=True,
                                                       hubble_units=False,
                                                       k_hunit=False)
        PKcamblin = results.get_matter_power_interpolator(nonlinear=False,
                                                          hubble_units=False,
                                                          k_hunit=False)
        self.s8 = results.get_sigma8()[-1]
        if self.verbose is True:
            print('# Sigma_8 = ', self.s8)

        zs_pow_int = np.repeat(self.zs, 12000)
        indi_k = np.append(np.append(np.linspace(0.0001, 0.22, 5000),
                                     np.linspace(0.22, 1.0, 5000)),
                           np.linspace(1.0, 500.0, 2000))
        k_list = np.tile(indi_k, len(self.zs))

        pk_list = []
        pk_lin_list = []

        for i in range(len(zs_pow_int)):
            pk = PKcamb.P(z=zs_pow_int[i], kh=k_list[i])
            pk_lin = PKcamblin.P(z=zs_pow_int[i], kh=k_list[i])
            pk_list.append(pk)
            pk_lin_list.append(pk_lin)

        xy = np.asarray([(i, j) for i, j in zip(k_list, zs_pow_int.tolist())])

        pk_k_chi_i = interpolate.LinearNDInterpolator(xy, np.array(pk_list),
                                                      fill_value=0.0)
        pk_lin_i = interpolate.LinearNDInterpolator(xy, np.array(pk_lin_list),
                                                  fill_value=0.0)
        self.power = pk_k_chi_i
        self.power_linear = pk_lin_i

        z_long = np.logspace(-3.0, 1.0, 100)
        D_list = []
        hub_list = []
        for rsh in z_long:
            D_list.append(
                np.sqrt(PKcamb.P(z=rsh, kh=0.1) / PKcamb.P(z=0.0, kh=0.1)))
            hub_list.append(results.hubble_parameter(rsh))
        e_list = self.H0 * copy.deepcopy(np.array(hub_list))
        self.D_growth = interpolate.InterpolatedUnivariateSpline(x=z_long,
                                                                 y=D_list,
                                                                 ext=0)
        self.Hubble_func = interpolate.InterpolatedUnivariateSpline(x=z_long,
                                                                 y=hub_list,
                                                                 ext=0)
        self.E_func = interpolate.InterpolatedUnivariateSpline(x=z_long,
                                                                 y=e_list,
                                                                 ext=0)
        return

    def kern_integral(self, z_dash, z, n_of_z):
        """
        Integrand of lensing kernel.

        Parameters
        ----------
        z_dash: float
            Dummy redshift variable being integrated over.
        z: float
            Redshift at which lensing kernel is being evaluated.
        n_of_z: function
            Galaxy distribution function of current bin.

        Returns
        -------
        float
            Value of integrand at current z_dash.
        """
        wint = n_of_z(z_dash) * (1.0 - (self.z_to_chi(z) /
                                    self.z_to_chi(z_dash)))
        return wint

    def kernel_gamma(self, z, bin_index):
        """
        Calculates value of lensing kernel at a given redshift.

        Parameters
        ----------
        z: float
            Redshift at which lensing kernel is being evaluated.
        bin_index: int
            Index of redshift bin being evaluated. Indices starts from 1.

        Returns
        -------
        float
            Value of kernel at current redshift.
        """
        cur_nz = self.n_funcs[int(bin_index-1)]

        if self.survey == 'photometric':
            W_val = (1.5 * (self.H0 / self.c_light) * self.om_m * (1.0 + z) * (
                    self.z_to_chi(z) /
                    (self.c_light / self.H0)) * integrate.quad(self.kern_integral,
                                               a=z, b=self.survey_max-0.1,
                                               args=(z, cur_nz))[0])
        else:
            bin_limits_list = [0.001, 0.568, 0.654, 0.723, 0.788, 0.851,
                               0.921, 0.999, 1.097, 1.243, 1.68]
            bin_min = bin_limits_list[bin_index-1]
            bin_max = bin_limits_list[bin_index]
            if z >= bin_max:
                W_val = 0.0
            elif z <= bin_min:
                W_val = (1.5 * (self.H0 / self.c_light) * self.om_m * (
                            1.0 + z) * (
                                 self.z_to_chi(z) /
                                 (self.c_light / self.H0)) *
                         integrate.quad(self.kern_integral,
                                        a=bin_min, b=bin_max,
                                        args=(z, cur_nz))[0])
            else:
                W_val = (1.5 * (self.H0 / self.c_light) * self.om_m * (
                        1.0 + z) * (
                                 self.z_to_chi(z) /
                                 (self.c_light / self.H0)) *
                         integrate.quad(self.kern_integral,
                                        a=z, b=bin_max,
                                        args=(z, cur_nz))[0])
        return W_val

    def kernel_k_dop_like(self, z, ell, bin_index):
        """
        Calculates value of the kappa-doppler-like kernel at a given redshift.

        Parameters
        ----------
        z: float
            Redshift at which lensing kernel is being evaluated.
        ell: float
            Ell-mode at which kernel is evaluated.
        bin_index: int
            Index of redshift bin being evaluated. Indices starts from 1.

        Returns
        -------
        float
            Value of kernel at current redshift.
        """
        cur_nz = self.n_funcs[int(bin_index - 1)](z)
        W_val = ((ell / ((ell + 0.5) ** 2.0)) - (1.0 / (ell + 1.5))) * \
                cur_nz * self.z_to_chi(z) * ((self.c_light) /
                                             ((self.z_to_chi(z) ** 2.0) *
                                              self.Hubble_func(z) *
                                              (1.0 / (1.0 + z))))
        return W_val

    def kernel_gamma_dop_like(self, z, bin_index):
        """
        Calculates value of the gamma-doppler-like kernel at a given redshift.

        Parameters
        ----------
        z: float
            Redshift at which lensing kernel is being evaluated.
        bin_index: int
            Index of redshift bin being evaluated. Indices starts from 1.

        Returns
        -------
        float
            Value of kernel at current redshift.
        """
        cur_nz = self.n_funcs[int(bin_index - 1)](z)
        prefac = 1.5 * ((self.om_m * (self.H0**2.0) / ((self.c_light)**2.0)))
        W_val = prefac * cur_nz
        return W_val

    def c_ell_integrand(self, z, W_i_z, W_j_z, ell):
        """
        Calculates the C_\ell integrand for any two bins supplied.

        Parameters
        ----------
        z: float
            Redshift at which integrand is being evaluated.
        W_i_z: float
           Value of kernel for bin i, at redshift z.
        W_j_z: float
           Value of kernel for bin j, at redshift z.
        ell: float
           \ell-mode at which the current C_\ell is being evaluated at.
        Returns
        -------
        float
            Value of C_\ell integrand at redshift z.
        """
        kern_mult = ((W_i_z * W_j_z) /
                     (self.Hubble_func(z) *
                      (self.z_to_chi(z)) ** 2.0))
        k = (ell + 0.5) / self.z_to_chi(z)
        return self.c_light * kern_mult * self.power(k, z)

    def c_Ig_integrand(self, z, W_i_z, W_j_z, n_i_z, n_j_z, ell):
        """
        Calculates the intrinsic alignment C_\ell^Ig integrand
        for any two bins supplied.

        Parameters
        ----------
        z: float
            Redshift at which integrand is being evaluated.
        W_i_z: float
           Value of kernel for bin i, at redshift z.
        W_j_z: float
           Value of kernel for bin j, at redshift z.
        n_i_z: float
            Value of galaxy density for bin i, at redshift z.
        n_j_z: float
            Value of galaxy density for bin j, at redshift z.
        ell: float
           \ell-mode at which the current C_\ell is being evaluated at.
        Returns
        -------
        float
            Value of C_\ell integrand at redshift z.
        """
        kern_mult = (((W_i_z * n_j_z / (self.c_light/self.Hubble_func(z))) +
                      (W_j_z * n_i_z / (self.c_light/self.Hubble_func(z)))) /
                     (self.Hubble_func(z) *
                      (self.z_to_chi(z)) ** 2.0))
        IA_factor = -1.0 * ((self.A_IA * self.C_IA * self.om_m)
                            /self.D_growth(z)) * (self.Hubble_func(z) /
                                                  self.c_light)
        k = (ell + 0.5) / self.z_to_chi(z)
        return self.c_light * kern_mult * IA_factor * self.power(k, z)

    def c_II_integrand(self, z, n_i_z, n_j_z, ell):
        """
        Calculates the intrinsic alignment C_\ell^II integrand
        for any two bins supplied.

        Parameters
        ----------
        z: float
            Redshift at which integrand is being evaluated.
        n_i_z: float
            Value of galaxy density for bin i, at redshift z.
        n_j_z: float
            Value of galaxy density for bin j, at redshift z.
        ell: float
           \ell-mode at which the current C_\ell is being evaluated at.
        Returns
        -------
        float
            Value of C_\ell integrand at redshift z.
        """
        kern_mult = ((n_i_z * n_j_z / ((self.c_light/self.Hubble_func(
            z))**2.0)) /
                     (self.Hubble_func(z) *
                      (self.z_to_chi(z)) ** 2.0))
        IA_factor = -1.0 * ((self.A_IA * self.C_IA * self.om_m)
                            /self.D_growth(z)) * (self.Hubble_func(z) /
                                                  self.c_light)
        k = (ell + 0.5) / self.z_to_chi(z)
        return self.c_light * kern_mult * (IA_factor**2.0) * self.power(k, z)

    def c_ell(self, ell, bin_i_index, bin_j_index):
        """
        Calculates the C_\ell for any two bins supplied.

        Parameters
        ----------
        ell: float
           \ell-mode at which the current C_\ell is being evaluated at.
        bin_i_index: int
            Index of bin i. Indices starts from 1.
        bin_j_index: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of C_\ell.
        """
        bin_i_kern = self.kern_list[bin_i_index-1]
        bin_j_kern = self.kern_list[bin_j_index-1]
        c_inps = []
        for z_ind in range(len(self.zs)):
            c_inps.append([self.zs[z_ind], bin_i_kern[z_ind],
                           bin_j_kern[z_ind], ell])
        p = Pool(self.proc)
        cs = p.starmap(self.c_ell_integrand, c_inps)
        p.close()
        p.join()
        int_c = integrate.trapz(np.array(cs), self.zs)
        return int_c

    def c_ell_Ig(self, ell, bin_i_index, bin_j_index):
        """
        Calculates the intrinsic alignment C_\ell^Ig for any two bins
        supplied.

        Parameters
        ----------
        ell: float
           \ell-mode at which the current C_\ell is being evaluated at.
        bin_i_index: int
            Index of bin i. Indices starts from 1.
        bin_j_index: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of C_\ell.
        """
        bin_i_kern = self.kern_list[bin_i_index-1]
        bin_j_kern = self.kern_list[bin_j_index-1]
        bin_i_dist = self.n_BNT_list[bin_i_index-1]
        bin_j_dist = self.n_BNT_list[bin_j_index-1]
        c_inps = []
        for z_ind in range(len(self.zs)):
            c_inps.append([self.zs[z_ind], bin_i_kern[z_ind],
                           bin_j_kern[z_ind], bin_i_dist[z_ind],
                           bin_j_dist[z_ind], ell])
        p = Pool(self.proc)
        cs = p.starmap(self.c_Ig_integrand, c_inps)
        p.close()
        p.join()
        int_c = integrate.trapz(np.array(cs), self.zs)
        return int_c

    def c_ell_II(self, ell, bin_i_index, bin_j_index):
        """
        Calculates the intrinsic alignment C_\ell^II for any two bins
        supplied.

        Parameters
        ----------
        ell: float
           \ell-mode at which the current C_\ell is being evaluated at.
        bin_i_index: int
            Index of bin i. Indices starts from 1.
        bin_j_index: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of C_\ell.
        """
        bin_i_dist = self.n_BNT_list[bin_i_index-1]
        bin_j_dist = self.n_BNT_list[bin_j_index-1]
        c_inps = []
        for z_ind in range(len(self.zs)):
            c_inps.append([self.zs[z_ind], bin_i_dist[z_ind],
                           bin_j_dist[z_ind], ell])
        p = Pool(self.proc)
        cs = p.starmap(self.c_II_integrand, c_inps)
        p.close()
        p.join()
        int_c = integrate.trapz(np.array(cs), self.zs)
        return int_c

    def compute_default_c_ell_matrix(self):
        """
        Compute 10x10 matrices of standard and BNT C_\ells.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        c_arr = np.zeros((200, 10, 10))
        c_arr_BNT = np.zeros((200, 10, 10))
        full_ells = np.logspace(1.0, 3.7, 200)

        if self.survey == 'photometric':
            for bin_i in range(1, 11):
                for bin_j in range(1, 11):
                    if bin_j >= bin_i:
                        if self.verbose is True:
                            print('# Computing bin i:', bin_i, ', bin j:',
                                  bin_j)
                        cur_cs_gg = []
                        cur_cs_Ig = []
                        cur_cs_II = []
                        for ell in self.t_ells:
                            c_gg = self.c_ell(ell=ell, bin_i_index=bin_i,
                                              bin_j_index=bin_j)
                            c_Ig = self.c_ell_Ig(ell=ell, bin_i_index=bin_i,
                                                 bin_j_index=bin_j)
                            c_II = self.c_ell_II(ell=ell, bin_i_index=bin_i,
                                                 bin_j_index=bin_j)
                            cur_cs_gg.append(c_gg)
                            cur_cs_Ig.append(c_Ig)
                            cur_cs_II.append(c_II)
                        if self.verbose:
                            print('#    Initial values computed.')
                        c_gg_obj = interpolate.InterpolatedUnivariateSpline(
                            self.t_ells, cur_cs_gg)
                        c_Ig_obj = interpolate.InterpolatedUnivariateSpline(
                            self.t_ells, cur_cs_Ig)
                        c_II_obj = interpolate.InterpolatedUnivariateSpline(
                            self.t_ells, cur_cs_II)
                        if self.verbose:
                            print('#    Interpolations computed.')
                        cs = c_gg_obj(full_ells) + c_Ig_obj(full_ells) + \
                             c_II_obj(full_ells)
                        c_arr[:, bin_i-1, bin_j-1] = cs
                        if bin_i != bin_j:
                            c_arr[:, bin_j - 1, bin_i - 1] = cs

            for lind in range(len(full_ells)):
                c_arr_BNT[lind] = np.matrix(self.BNT_mat) * \
                                   np.matrix(c_arr[lind]) * \
                                   np.matrix(self.BNT_mat).T

            if self.verbose is True:
                print('# Computed C_ell matrix.')
            shot_noise = (0.3 ** 2.0) / ((30.0 * 11818102.86) / 10.0)
            shot_noise_mat = np.matrix(shot_noise * np.identity(10))
            shot_noise_BNT_mat = np.matrix(self.BNT_mat) * shot_noise_mat * \
                                 np.matrix(self.BNT_mat).T
            c_arr[:] += shot_noise_mat
            c_arr_BNT[:] += shot_noise_BNT_mat
            if self.verbose is True:
                print('# Computed and added shot noise.')
        elif self.survey == 'kinematic':
            for bin_i in range(1, 11):
                for bin_j in range(1, 11):
                    if bin_j >= bin_i:
                        if self.verbose is True:
                            print('# Computing bin i:', bin_i, ', bin j:',
                                  bin_j)
                        cur_cs_gg = []
                        for ell in self.t_ells:
                            c_gg = self.c_ell(ell=ell, bin_i_index=bin_i,
                                              bin_j_index=bin_j)
                            cur_cs_gg.append(c_gg)
                        if self.verbose:
                            print('#    Initial values computed.')
                        c_gg_obj = interpolate.InterpolatedUnivariateSpline(
                            self.t_ells, cur_cs_gg)
                        if self.verbose:
                            print('#    Interpolations computed.')
                        cs = c_gg_obj(full_ells)
                        c_arr[:, bin_i - 1, bin_j - 1] = cs
                        if bin_i != bin_j:
                            c_arr[:, bin_j - 1, bin_i - 1] = cs

            for lind in range(len(full_ells)):
                c_arr_BNT[lind] = np.matrix(self.BNT_mat) * \
                                  np.matrix(c_arr[lind]) * \
                                  np.matrix(self.BNT_mat).T

            if self.verbose is True:
                print('# Computed C_ell matrix.')
            shot_noise = (0.021 ** 2.0) / ((1.1 * 11818102.86) / 10.0)
            shot_noise_mat = np.matrix(shot_noise * np.identity(10))
            shot_noise_BNT_mat = np.matrix(self.BNT_mat) * shot_noise_mat * \
                                 np.matrix(self.BNT_mat).T
            c_arr[:] += shot_noise_mat
            c_arr_BNT[:] += shot_noise_BNT_mat
            if self.verbose is True:
                print('# Computed and added shot noise.')

        return c_arr, c_arr_BNT

    def bl_base_gen(self):
        """
        Compute and save matter bispectrum using Bihalofit code for future
        computation.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        test_ls = self.t_ells
        vr = []
        for ell in test_ls:
            vr.append(self.bf_l2_wrapper(ell))

        vr = np.array(vr)
        np.save('./temp_io/bispec_base.npy', vr)
        if self.verbose is True:
            print('# Computed matter bispectrum from bihalofit code.')
        print(vr.shape)
        return

    def bf_l2_wrapper(self, l1):
        """
        Compute and save matter bispectrum using Bihalofit code for future
        computation at l2 steps that will be used in reduced shear correction
        calculation.

        Parameters
        ----------
        l1: float
            First side of \ell triangle at which the bispectrum is to be
            evaluated.

        Returns
        -------
        array
            Matter bispectrum values, integrated over angle at required
            comoving distance and \ell_2 steps.
        """
        vals = []
        if l1 >= 51.0:
            ls = np.append(np.linspace(0.001, l1 - 50, 100, endpoint=False),
                           np.linspace(l1 + 50, 1.0e5, 100))

        else:
            ls = np.append(np.linspace(0.001, l1 - 5, 100, endpoint=False),
                           np.linspace(l1 + 50, 1.0e5, 100))

        for i in ls:
            vals.append([l1, i])

        p = Pool(self.proc)

        returnvals = p.starmap(self.bf_chi_wrapper, vals)
        p.close()
        p.join()
        returnvals = np.array(returnvals)
        return returnvals

    def bf_chi_wrapper(self, l1, l2):
        """
        Compute and save matter bispectrum using Bihalofit code for future
        computation at comoving distance steps that will be used in
        reduced shear correction
        calculation.

        Parameters
        ----------
        l1: float
            First side of \ell triangle at which the bispectrum is to be
            evaluated.
        l2: float
            Second side of \ell triangle at which the bispectrum is to be
            evaluated.

        Returns
        -------
        array
            Matter bispectrum values, integrated over angle at required
            comoving distance steps.
        """
        chis = []
        for rst in np.arange(0.001, 2.5, 0.1):
            chis.append(self.z_to_chi(rst))
        fins = []
        for item in chis:
            calc = self.B_mat_hf(l1, l2, item)
            fins.append(calc)
        fins = np.array(fins)
        return fins

    def B_mat_hf(self, l1, l2, chi):
        """
        Compute and save matter bispectrum using Bihalofit code for future
        computation, integrating over angle, that will be used in
        reduced shear correction
        calculation.

        Parameters
        ----------
        l1: float
            First side of \ell triangle at which the bispectrum is to be
            evaluated.
        l2: float
            Second side of \ell triangle at which the bispectrum is to be
            evaluated.
        chi: float
            Comoving distance.
        Returns
        -------
        float
            Matter bispectrum value, integrated over angle at required
            steps.
        """
        in_vals = np.append(np.linspace(0.0, np.pi, 50, endpoint=False),
                            np.linspace(np.pi, 2.0 * np.pi, 50, endpoint=True))

        b_empty = []
        for item in in_vals:
            b_empty.append(self.B_integrand_angle(phi=item, l1=l1, l2=l2,
                                                  chi=chi))

        b_ret = integrate.trapz(b_empty, in_vals) / ((2.0 * np.pi) ** 2.0)
        return b_ret

    def B_integrand_angle(self, phi, l1, l2, chi):
        """
        Compute integrand of angular integration over bispectrum for reduced
        shear correction.

        Parameters
        ----------
        phi: float
            Angle at which to evaluate integrand
        l1: float
            First side of \ell triangle at which the bispectrum is to be
            evaluated.
        l2: float
            Second side of \ell triangle at which the bispectrum is to be
            evaluated.
        chi: float
            Comoving distance.
        Returns
        -------
        float
            Value of integrand.
        """
        k1 = (l1 + 0.5) / chi
        k2 = (l2 + 0.5) / chi

        l3 = np.sqrt(l1 ** 2.0 + l2 ** 2.0 + 2.0 * l1 * l2 * np.cos(phi))
        k3 = (l3 + 0.5) / chi

        if chi == 0.0:
            z = 0.0
        else:
            z = self.chi_to_z(chi)

        b_val = np.cos(2.0*phi) * (self.bihalofit_matter(k1=k1, k2=k2, k3=k3,
                                                        z=z))
        return b_val

    def bihalofit_matter(self, k1, k2, k3, z):
        """
        Compute Matter bispectrum from bihalofit C code.

        Parameters
        ----------
        k1: float
            First side of k triangle to evaluate bispectrum.
        k2: float
            Seond side of k triangle to evaluate bispectrum.
        k3: float
            Third side of k triangle to evaluate bispectrum.
        z: float
            Redshift at which to evaluate bispectrum.
        Returns
        -------
        float
            Value of matter bispectrum at k1, k2, k3, z.
        """
        if np.max([k1, k2, k3]) < k1+k2+k3-np.max([k1, k2, k3]):
            h = self.h
            sigma8 = self.s8
            omb = self.om_b
            omc = self.om_cdm
            ns = abs(self.ns)
            w = self.w0
            ell_string = " ".join(
                [str(h), str(sigma8), str(omb), str(omc), str(ns), str(w),
                 str(k1/h), str(k2/h), str(k3/h), str(z)])
            pstemp = subprocess.Popen('./bihalofit/a.out',
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      close_fds=True)
            restemp = pstemp.communicate(ell_string.encode('utf-8'))
            pstemp.stdout.close()
            pstemp.stdin.close()
            pstemp.kill()
            instringtmp = restemp[0].decode("utf-8").split()
            if instringtmp[13] == 'not':
                print(instringtmp)
            solv = float(instringtmp[13])
        else:
            solv = 0.0
        return solv

    def del_C_RS(self, l, lind, bin_i_ind, bin_j_ind):
        """
        Calculates the reduced shear correction for specified bins, at given
        \ell.

        Parameters
        ----------
        l: float
            \ell-mode at which to compute correction.
        lind: float
            Index corresponding to \ell-mode in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of reduced shear correction.
        """
        if self.verbose is True:
            print('#    Calculating RS correction for ell = ', l)
        vals = []
        if l >= 51.0:
            ls = np.append(np.linspace(0.001, l - 50, 100, endpoint=False),
                           np.linspace(l + 50, 1.0e5, 100))

        else:
            ls = np.append(np.linspace(0.001, l - 5, 100, endpoint=False),
                           np.linspace(l + 50, 1.0e5, 100))

        for i in range(len(ls)):
            vals.append([ls[i], lind, i, bin_i_ind, bin_j_ind])

        p = Pool(self.proc)

        returnvals = p.starmap(self.del_c_b_int, vals)
        p.close()
        p.join()
        returnvals = np.array(returnvals)

        smint1 = 2.0 * integrate.trapz(returnvals, ls)

        return smint1

    def del_C_doppler(self, l, lind, bin_i_ind, bin_j_ind):
        """
        Calculates the doppler correction for specified bins, at given
        \ell.

        Parameters
        ----------
        l: float
            \ell-mode at which to compute correction.
        lind: float
            Index corresponding to \ell-mode in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of reduced shear correction.
        """
        if self.verbose is True:
            print('#    Calculating Doppler correction for ell = ', l)
        vals = []
        if l >= 51.0:
            ls = np.append(np.linspace(0.001, l - 50, 100, endpoint=False),
                           np.linspace(l + 50, 1.0e5, 100))

        else:
            ls = np.append(np.linspace(0.001, l - 5, 100, endpoint=False),
                           np.linspace(l + 50, 1.0e5, 100))

        for i in range(len(ls)):
            vals.append([ls[i], lind, i, bin_i_ind, bin_j_ind, 'Doppler'])

        p = Pool(self.proc)

        returnvals = p.starmap(self.del_c_b_int, vals)
        p.close()
        p.join()
        returnvals = np.array(returnvals)

        smint1 = 2.0 * integrate.trapz(returnvals, ls)

        return smint1

    def del_C_IA_doppler(self, l, lind, bin_i_ind, bin_j_ind):
        """
        Calculates the doppler-IA correction for specified bins, at given
        \ell.

        Parameters
        ----------
        l: float
            \ell-mode at which to compute correction.
        lind: float
            Index corresponding to \ell-mode in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of reduced shear correction.
        """
        if self.verbose is True:
            print('#    Calculating Doppler correction for ell = ', l)
        vals = []
        if l >= 51.0:
            ls = np.append(np.linspace(0.001, l - 50, 100, endpoint=False),
                           np.linspace(l + 50, 1.0e5, 100))

        else:
            ls = np.append(np.linspace(0.001, l - 5, 100, endpoint=False),
                           np.linspace(l + 50, 1.0e5, 100))

        for i in range(len(ls)):
            vals.append([ls[i], l, bin_i_ind, bin_j_ind])

        p = Pool(self.proc)

        returnvals = p.starmap(self.del_c_b_int_IA, vals)
        p.close()
        p.join()
        returnvals = np.array(returnvals)

        smint1 = 2.0 * integrate.trapz(returnvals, ls)
        return smint1

    def del_c_b_int(self, l_prime, l1ind, l2ind,
                    bin_i_ind, bin_j_ind, correction_type='RS'):
        """
        Calculate integrand over \ell for the reduced shear or Doppler
        correction.

        Parameters
        ----------
        l_prime: float
            Dummy integration variable for \ell.
        l1ind: float
            Index corresponding to \ell_1 in precomputed bispectrum array.
        l2ind: float
            Index corresponding to \ell_2 in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of reduced shear correction integrand.
        """
        if correction_type == 'RS':
            val = l_prime * self.B_ij(l1ind, l2ind,
                                      bin_i_ind, bin_j_ind)
        elif correction_type == 'Doppler':
            val = l_prime * self.B_ij_doppler(l1ind, l2ind,
                                              bin_i_ind, bin_j_ind, l_prime)
        else:
            raise Exception('# Correction type must be RS or Doppler.')
        return val

    def del_c_b_int_IA(self, l_prime, const_l, bin_i_ind, bin_j_ind):
        """
        Calculate integrand over \ell for the Doppler-IA correction.

        Parameters
        ----------
        l_prime: float
            Dummy integration variable for \ell.
        const_l: float
            \ell-mode at which correction is being evaluated at.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of Doppler-IA correction integrand.
        """

        val1 = l_prime * self.B_ij_IA(const_l, l_prime, bin_i_ind, bin_j_ind)
        return val1

    def B_ij_IA(self, l1, l2, bin_i, bin_j):
        """
        Calculates the IA-Doppler bispectrum.

        Parameters
        ----------
        l1: float
            First \ell side of bispectrum triangle.
        l2: float
            Second \ell side of bispectrum triangle.
        bin_i: int
            Index of bin i. Indices starts from 1.
        bin_j: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of IA-Doppler bispectrum.
        """

        chis = []
        for rst in np.arange(0.001, 2.5, 0.1):
            chis.append(self.z_to_chi(rst))
        fins = []
        for item in range(len(chis)):
            calc = self.B_ij_IA_integrand(chis[item], l1, l2, bin_i, bin_j)
            fins.append(calc)

        int_val = 0.5 * integrate.trapz(fins, chis)
        return int_val

    def B_ij_IA_integrand(self, chi, l1, l2, bin_i, bin_j):
        """
        Calculates integrand for the IA-Doppler bispectrum.

        Parameters
        ----------
        chi: float
            Comoving distance to be integrated over.
        l1: float
            First \ell side of bispectrum triangle.
        l2: float
            Second \ell side of bispectrum triangle.
        bin_i: int
            Index of bin i. Indices starts from 1.
        bin_j: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of IA-Doppler bispectrum integrand.
        """
        W_i_kv = self.kernel_k_dop_like(self.chi_to_z(chi), l2, bin_i)
        W_j_kv = self.kernel_k_dop_like(self.chi_to_z(chi), l2, bin_j)

        W_i_gv = self.kernel_gamma_dop_like(self.chi_to_z(chi), bin_i)
        W_j_gv = self.kernel_gamma_dop_like(self.chi_to_z(chi), bin_j)

        dz_dx = misc.derivative(self.chi_to_z, x0=chi)
        n_i = self.n_funcs[int(bin_i-1)](self.chi_to_z(chi)) * dz_dx
        n_j = self.n_funcs[int(bin_j - 1)](self.chi_to_z(chi)) * dz_dx

        val = (1.0 / (chi ** 4.0)) * \
              ((W_i_kv * W_i_gv * n_j) + (W_j_kv * W_j_gv * n_i)) * \
              self.B_IA_gen(l1, l2, chi)
        return val

    def B_IA_gen(self, l1, l2, chi):
        """
        Calculates the IA bispectrum.

        Parameters
        ----------
        l1: float
            First \ell side of bispectrum triangle.
        l2: float
            Second \ell side of bispectrum triangle.
        chi: float
            Comoving distance.
        Returns
        -------
        float
            Value of IA bispectrum.
        """

        k1 = (l1 + 0.5)/chi
        k2 = (l2 + 0.5)/chi
        pow1 = (-self.A_IA * self.C_IA * (self.om_m /
                                          self.D_growth(self.chi_to_z(chi))))\
               * self.power(k1, self.chi_to_z(chi))
        pow2 = (-self.A_IA * self.C_IA * (self.om_m /
                                          self.D_growth(self.chi_to_z(chi))))\
               * self.power(k2, self.chi_to_z(chi))

        knl = self.skdict['{:d}'.format(int(10.0 *
                                            (self.chi_to_z(chi) + 1)))][1]

        in_vals = np.append(np.linspace(0.0, np.pi, 50, endpoint=False),
                            np.linspace(np.pi, 2.0 * np.pi, 50, endpoint=True))

        F13_vals = []
        F23_vals = []
        for item in in_vals:
            F13_vals.append(
                self.F_eff13_gen(item, k1, knl, l1, l2, chi))
            F23_vals.append(
                self.F_eff23_gen(item, k2, knl, l1, l2, chi))

        B = ((2.0 * (2.0 / 7.0) * (np.pi / 2.0) * pow1
              * pow2) + (2.0 * integrate.trapz(F13_vals, in_vals)) +
             (2.0 * integrate.trapz(F23_vals, in_vals))) / (
                        (2.0 * np.pi) ** 2.0)
        return B

    def F_eff13_gen(self, phi, k1, knl, l1, l2, chi):
        """
        Evaluation of F_eff fitting function from Scoccimaro, Couchman 2001
        (https://doi.org/10.1046/j.1365-8711.2001.04281.x)
        in the case of l, -l-'.

        Parameters
        ----------
        phi: float
            Angle of l2 w.r.t to l1.
        k1: float
            k-mode corresponding to l1.
        s8: float
            Sigma_8 cosmological parameter; amplitude of matter power spectrum.
        knl: float
            Scale of non-linearities.
        l1: float
            First \ell side of bispectrum triangle.
        l2: float
            Second \ell side of bispectrum triangle.
        chi: float
            Comoving distance.

        Returns
        -------
        float
            Value of fitting function.
        """

        pow1 = self.power(k1, chi)

        l3 = np.sqrt(l1 ** 2.0 + l2 ** 2.0 + 2.0 * l1 * l2 * np.cos(phi))
        k3 = (l3 + 0.5) / chi

        pow3 = (-self.A_IA * self.C_IA * (self.om_m / self.D_growth(
            self.chi_to_z(chi)))) * self.power(k3, self.chi_to_z(chi))

        ang_3 = np.pi + np.arctan2(l2 * np.sin(phi), l1 + l2 * np.cos(phi))

        val = (((5.0 / 7.0) * self.a_bispec_IA(k1, knl) * self.a_bispec_IA(k3, knl)) +
               (0.5 * (k1 / k3 + k3 / k1) * self.b_bispec_IA(k1, knl) * self.b_bispec_IA(
                   k3, knl) * np.cos(ang_3)) +
               ((2.0 / 7.0) * self.c_bispec_IA(k1, knl) * self.c_bispec_IA(k3, knl) * (
                           np.cos(ang_3) ** 2.0))) * np.cos(2.0 * phi) * \
              pow1 * pow3

        return val

    def F_eff23_gen(self, phi, k2, knl, l1, l2, chi):
        """
        Evaluation of bispectrum fitting function F_eff from Scoccimaro,
        Couchman 2001 (https://doi.org/10.1046/j.1365-8711.2001.04281.x)
        in the case of l', -l-'.

        Parameters
        ----------
        phi: float
            Angle of l2 w.r.t to l1.
        k2: float
            k-mode corresponding to l2.
        s8: float
            Sigma_8 cosmological parameter; amplitude of matter power spectrum.
        knl: float
            Scale of non-linearities.
        l1: float
            First \ell side of bispectrum triangle.
        l2: float
            Second \ell side of bispectrum triangle.
        chi: float
            Comoving distance.

        Returns
        -------
        float
            Value of fitting function.
        """

        pow2 = self.power(k2, chi)

        l3 = np.sqrt(l1 ** 2.0 + l2 ** 2.0 + 2.0 * l1 * l2 * np.cos(phi))

        k3 = (l3 + 0.5) / chi

        pow3 = (-self.A_IA * self.C_IA *
                (self.om_m/self.D_growth(self.chi_to_z(chi))))\
               * self.power(k3, self.chi_to_z(chi))

        ang_3 = np.pi + np.arctan2(l2 * np.sin(phi), l1 + l2 * np.cos(phi))

        val = (((5.0/7.0) * self.a_bispec_IA(k2, knl) * self.a_bispec_IA(k3, knl)) +
               (0.5 * (k3/k2 + k2/k3) * self.b_bispec_IA(k3, knl) * self.b_bispec_IA(k2, knl) * np.cos(phi+np.pi-ang_3)) +
               (2.0 / 7.0)*(self.c_bispec_IA(k3, knl) * self.c_bispec_IA(k2, knl) * (np.cos(phi+np.pi-ang_3)**2.0))) * np.cos(2.0*phi) * \
              pow2 * pow3

        return val

    def a_bispec_IA(self, k, knl):
        """
        Evaluation of bispectrum fitting function a from Scoccimaro,
        Couchman 2001 (https://doi.org/10.1046/j.1365-8711.2001.04281.x)

        Parameters
        ----------
        k: float
            k-mode at which to evaluate fitting function.
        knl: float
            Scale of non-linearities.

        Returns
        -------
        float
            Value of fitting function.
        """
        q = k / (knl)
        Q3 = (4.0 - (2.0 ** self.ns)) / (1.0 + (2.0 ** (self.ns + 1.0)))

        a_val = (1.0 + (self.s8 ** -0.2) * np.sqrt(0.7 * Q3) * (q / 4.0) ** (
                    self.ns + 3.5)) / (1.0 + (q / 4.0) ** (self.ns + 3.5))
        return a_val

    def b_bispec_IA(self, k, knl):
        """
        Evaluation of bispectrum fitting function b from Scoccimaro,
        Couchman 2001 (https://doi.org/10.1046/j.1365-8711.2001.04281.x)

        Parameters
        ----------
        k: float
            k-mode at which to evaluate fitting function.
        knl: float
            Scale of non-linearities.

        Returns
        -------
        float
            Value of fitting function.
        """
        q = k / (knl)

        b_val = (1.0 + (0.4 * (self.ns + 3.0) * q ** (self.ns + 3.0))) / (
                    1.0 + q ** (self.ns + 3.5))
        return b_val

    def c_bispec_IA(self, k, knl):
        """
        Evaluation of bispectrum fitting function c from Scoccimaro,
        Couchman 2001 (https://doi.org/10.1046/j.1365-8711.2001.04281.x)

        Parameters
        ----------
        k: float
            k-mode at which to evaluate fitting function.
        knl: float
            Scale of non-linearities.

        Returns
        -------
        float
            Value of fitting function.
        """
        q = k / (knl)

        if q == 0.0:
            c_val = 0.0
        else:
            c_val = (1.0 + (4.5 / ((1.5 + (self.ns + 3.0) ** 4.0) * (
                        (2.0 * q) ** (self.ns + 3.0))))) / (
                                1.0 + (2.0 * q) ** (self.ns + 3.5))
        return c_val

    def B_ij(self, l1ind, l2ind, bin_i_ind, bin_j_ind):
        """
        Calculate the lensing bispectrum for bins i and j,
        at given \ell, \ell_prime.

        Parameters
        ----------
        l1ind: float
            Index corresponding to \ell_1 in precomputed bispectrum array.
        l2ind: float
            Index corresponding to \ell_2 in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of lensing bispectrum.
        """

        chis = []
        for rst in np.arange(0.001, 2.5, 0.1):
            chis.append(self.z_to_chi(rst))
        fins = []
        for item in range(len(chis)):
            calc = self.B_ij_integrand(chis[item], l1ind, l2ind, item,
                                       bin_i_ind, bin_j_ind)
            fins.append(calc)

        int_val = 0.5 * integrate.trapz(fins, chis)
        return int_val

    def B_ij_integrand(self, chi, l1ind, l2ind, chiind,
                       bin_i_ind, bin_j_ind):
        """
        Calculate the integrand of the lensing bispectrum for bins i and j,
        at given \ell, \ell_prime.

        Parameters
        ----------
        chi: float
            Comoving distance, to be integrated over.
        l1ind: int
            Index corresponding to \ell_1 in precomputed bispectrum array.
        l2ind: int
            Index corresponding to \ell_2 in precomputed bispectrum array.
        chiind: int
            Index corresponding to chi in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        Returns
        -------
        float
            Value of lensing bispectrum integrand.
        """
        W_i = self.kernel_gamma(self.chi_to_z(chi), bin_index=bin_i_ind)
        W_j = self.kernel_gamma(self.chi_to_z(chi), bin_index=bin_j_ind)

        val = (1.0 / (chi ** 4.0)) * W_i * W_j * (
                W_i + W_j) * self.b_mat_ld[l1ind, l2ind, chiind]
        return val

    def B_ij_doppler(self, l1ind, l2ind, bin_i_ind, bin_j_ind, l_prime):
        """
        Calculate the Doppler correction bispectrum for bins i and j,
        at given \ell, \ell_prime.

        Parameters
        ----------
        l1ind: float
            Index corresponding to \ell_1 in precomputed bispectrum array.
        l2ind: float
            Index corresponding to \ell_2 in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        l_prime: float
            \ell-mode being integrated over in correction calculation.
        Returns
        -------
        float
            Value of Doppler bispectrum.
        """

        chis = []
        for rst in np.arange(0.001, 2.5, 0.1):
            chis.append(self.z_to_chi(rst))
        fins = []
        for item in range(len(chis)):
            calc = self.B_ij_integrand_doppler(chis[item], l1ind, l2ind, item,
                                               bin_i_ind, bin_j_ind, l_prime)
            fins.append(calc)

        int_val = 0.5 * integrate.trapz(fins, chis)
        return int_val

    def B_ij_integrand_doppler(self, chi, l1ind, l2ind, chiind,
                               bin_i_ind, bin_j_ind, l_prime):
        """
        Calculate the integrand of the Doppler-correction bispectrum for bins
        i and j, at given \ell, \ell_prime.

        Parameters
        ----------
        chi: float
            Comoving distance, to be integrated over.
        l1ind: int
            Index corresponding to \ell_1 in precomputed bispectrum array.
        l2ind: int
            Index corresponding to \ell_2 in precomputed bispectrum array.
        chiind: int
            Index corresponding to chi in precomputed bispectrum array.
        bin_i_ind: int
            Index of bin i. Indices starts from 1.
        bin_j_ind: int
            Index of bin j. Indices starts from 1.
        l_prime: float
            \ell-mode being integrated over in correction calculation.
        Returns
        -------
        float
            Value of Doppler bispectrum integrand.
        """
        W_i = self.kernel_gamma(self.chi_to_z(chi), bin_index=bin_i_ind)
        W_j = self.kernel_gamma(self.chi_to_z(chi), bin_index=bin_j_ind)

        W_i_kv = self.kernel_k_dop_like(self.chi_to_z(chi), l_prime, bin_i_ind)
        W_j_kv = self.kernel_k_dop_like(self.chi_to_z(chi), l_prime, bin_j_ind)

        W_i_gv = self.kernel_gamma_dop_like(self.chi_to_z(chi), bin_i_ind)
        W_j_gv = self.kernel_gamma_dop_like(self.chi_to_z(chi), bin_j_ind)

        val = (1.0 / (chi ** 4.0)) * \
              ((W_i_kv * W_i_gv * W_j) + (W_j_kv * W_j_gv * W_i)) * \
              self.b_mat_ld[l1ind, l2ind, chiind]
        return val

    def compute_correction_matrix(self, correction_type='RS'):
        """
        Calculates the 10x10 matrix of requested corrections for each bin,
        for each requested \ell-mode.

        Parameters
        ----------
        correction_type: str
            Type of correction to compute: 'RS' for reduced shear, 'Doppler'
            for basic Doppler-shft correction, and 'IA-Doppler' for the
            Doppler-IA correction term.

        Returns
        -------
        array, array
            Standard and BNT transformed reduced shear correction matrices.
        """
        if correction_type == 'RS':
            d_func = self.del_C_RS
        elif correction_type == 'Doppler':
            d_func = self.del_C_doppler
        elif correction_type == 'IA-Doppler':
            d_func = self.del_C_IA_doppler
            with open('./temp_io/m_s8_knl.pkl', 'rb') as inth:
                sk_dict = pickle.load(inth)
            self.skdict = sk_dict
        else:
            raise Exception('# Correction type must be RS or Doppler.')

        dc_arr = np.zeros((200, 10, 10))
        dc_arr_BNT = np.zeros((200, 10, 10))
        full_ells = np.logspace(1.0, 3.7, 200)

        for bin_i in range(1, 11):
            for bin_j in range(1, 11):
                if bin_j >= bin_i:
                    if self.verbose is True:
                        print('# Computing {:s} corr bin i:'.format(
                            correction_type), bin_i, ', bin j:', bin_j)
                    cur_dcs = []
                    for ell_ind in range(len(self.t_ells)):
                        dc = d_func(l=self.t_ells[ell_ind], lind=ell_ind,
                                           bin_i_ind=bin_i, bin_j_ind=bin_j)
                        cur_dcs.append(dc)
                    if self.verbose:
                        print('#    Initial values computed.')
                    dc_obj = interpolate.InterpolatedUnivariateSpline(
                        self.t_ells, cur_dcs)
                    if self.verbose:
                        print('#    Interpolations computed.')

                    cs = dc_obj(full_ells)

                    dc_arr[:, bin_i - 1, bin_j - 1] = cs
                    if bin_i != bin_j:
                        dc_arr[:, bin_j - 1, bin_i - 1] = cs

        for lind in range(len(self.t_ells)):
            dc_arr_BNT[lind] = np.matrix(self.BNT_mat) * \
                               np.matrix(dc_arr[lind]) * \
                               np.matrix(self.BNT_mat).T

        if self.verbose is True:
            print('# Computed del_C_ell matrix.')
        return dc_arr, dc_arr_BNT


class Fishercalc:
    """
    Class to carry out Fisher matrix and Bias analysis.
    """
    def __init__(self, def_ells=np.logspace(1.0, 3.7, 200),
                 survey='photometric',
                 kcut=None, lcut=None, verbose=True):
        """

        Parameters
        ----------
        def_ells: array
            \ell-modes at which quantities are computed at.
        survey: str
            Type of survey to compute Fisher for: 'photometric' for Stage IV
            Euclid-like survey, and 'kinematic' for hypothetical Tully-Fisher
            survey.
        kcut: float
            k-scale at which to cut BNT spectrum (Mpc^{-1}). Either an l-cut
            or k-cut can be specified, not both
        lcut: float
            l-mode at which to cut BNT spectrum. Either an l-cut or k-cut
            csn be specified, not both.
        verbose: bool
            Option to print verbose output.
        """
        self.ells = def_ells
        self.kcut = kcut
        self.lcut = lcut
        self.survey = survey
        self.verbose = verbose

        if self.survey not in ['photometric', 'kinematic']:
            raise Exception('Survey must be either photometric - for Stage IV'
                            'or kinematic for hypothetical kinematic lensing'
                            'survey.')

        if self.kcut is not None and self.lcut is not None:
            raise Exception('Either an l-cut or k-cut must be specified, not'
                            ' both.')

        chis_interp = np.load('./temp_io/fsh_comov_vals.npy')
        self.z_to_chi = interpolate.InterpolatedUnivariateSpline(
            np.arange(0.001, 4.0, 0.1), chis_interp)
        if self.survey == 'photometric':
            self.med_chis = np.load('./temp_io/med_list_phot.npy')
        else:
            self.med_chis = np.load('./temp_io/med_list_kin.npy')
        if self.kcut is not None:
            self.bin_cuts = self.kcut * np.array(self.med_chis)
            self.ind_cuts = []
            for cut in self.bin_cuts:
                sim_ind = self.find_nearest(self.ells, cut)
                self.ind_cuts.append(sim_ind)
        elif self.lcut is not None:
            self.ind_cuts = self.find_nearest(self.ells, self.lcut)
        else:
            self.bin_cuts = None
            self.ind_cuts = None
        return

    def inv(self, matrix):
        """
        Calculates the inverse of an input matrix. (Greater tolerance than
        np.matrix.inverse)

        Parameters
        ----------
        matrix: array
            Matrix to be inverted.

        Returns
        -------
        array
            Inverse of input matrix.
        """
        ax1, ax2 = matrix.shape
        if ax1 != ax2:
            raise Exception("Matrix must be square in order to invert.")

        int_m = np.eye(ax1, ax1)
        return np.linalg.lstsq(matrix, int_m, rcond=None)[0]

    def find_nearest(self, inp_array, value):
        """
        Finds the index of the item in inp_array that is the closest match to
        value.

        Parameters
        ----------
        inp_array: array
            Array to search.
        value: float
            Value to find nearest item to in array.
        Returns
        -------
        int
            Index of closest match.
        """
        inp_array = np.array(inp_array)
        idx = (np.abs(inp_array - value)).argmin()
        return idx

    def apply_lcut(self, full_matrix):
        """
        Applies l-cut to array of C_\ells, derivatives, or corrections.

        Parameters
        ----------
        full_matrix: array
            Array to apply l-cut to.

        Returns
        -------
        array
            Array after l-cut has been applied.
        """
        return full_matrix[:self.ind_cuts, :, :]

    def apply_kcut(self, full_matrix):
        """
        Applies k-cut to array of C_\ells, derivatives, or corrections.

        Parameters
        ----------
        full_matrix: array
            Array to apply k-cut to.

        Returns
        -------
        array
            Array after k-cut has been applied.
        """
        cut_list = []
        for ell_ind in range(len(self.ells)):
            if self.ells[ell_ind] >= np.min(self.bin_cuts):
                sim_ind = self.find_nearest(self.bin_cuts, self.ells[ell_ind])
                if self.ells[ell_ind] <= self.bin_cuts[sim_ind]:
                    cut_list.append(full_matrix[ell_ind][(sim_ind+1):,
                                    (sim_ind+1):])
                else:
                    cut_list.append(full_matrix[ell_ind][sim_ind:, sim_ind:])
            else:
                cut_list.append(full_matrix[ell_ind])
        return cut_list

    def in_fish_sum(self, lind, d_al, d_bet, inv_mat):
        """
        Computes the trace of the product of the angular power spectrum and its
        derivatives, at a given \ell-mode, as required in the Fisher matrix
        calculation.

        Parameters
        ----------
        lind: int
            Index of current \ell-mode within self.ells.
        d_al: array
            Matrix of C_\ell derivatives w.r.t parameter alpha.
        d_bet: array
            Matrix of C_\ell derivatives w.r.t parameter beta.
        inv_mat: array
            Inverse C_\ell matrix.

        Returns
        -------
        float
            Trace of product of matrices.
        """
        new_ls = self.ells
        if lind == 0:
            del_l = 10.0
        else:
            del_l = new_ls[lind] - new_ls[lind - 1]

        f_val_l = inv_mat * d_al * inv_mat * d_bet
        return f_val_l.trace()[0, 0] * (new_ls[lind] + 0.5) * del_l

    def fisher(self, cat='def'):
        """
        Computes the Fisher matrix for either the standard or BNT transformed
        C_\ells.

        Parameters
        ----------
        cat: str
            Category of Fisher matrix to compute. Either 'def' for the
            standard C_\ells, or 'BNT' for the BNT transformed C_\ells.
        Returns
        -------
        array
            Fisher matrix.
        """

        if self.survey == 'photometric':
            f_sky = 15000 * (np.pi / 180) ** 2 / (4 * np.pi)

            if cat=='def':
                cls_mat = np.load('./c_ells_default/cls_fiducial.npy')
                dom_mat = np.load('./c_ells_default/dcls_default_dom.npy')
                dob_mat = np.load('./c_ells_default/dcls_default_dob.npy')
                dh_mat = np.load('./c_ells_default/dcls_default_dh.npy')
                dns_mat = np.load('./c_ells_default/dcls_default_dns.npy')
                dw0_mat = np.load('./c_ells_default/dcls_default_dw0.npy')
                dwa_mat = np.load('./c_ells_default/dcls_default_dwa.npy')
                ds8_mat = np.load('./c_ells_default/dcls_default_ds8.npy')
                daia_mat = np.load('./c_ells_default/dcls_default_daia.npy')
            elif cat=='BNT':
                cls_mat = np.load('./c_ells_BNT/cls_BNT_fiducial.npy')
                dom_mat = np.load('./c_ells_BNT/dcls_BNT_dom.npy')
                dob_mat = np.load('./c_ells_BNT/dcls_BNT_dob.npy')
                dh_mat = np.load('./c_ells_BNT/dcls_BNT_dh.npy')
                dns_mat = np.load('./c_ells_BNT/dcls_BNT_dns.npy')
                dw0_mat = np.load('./c_ells_BNT/dcls_BNT_dw0.npy')
                dwa_mat = np.load('./c_ells_BNT/dcls_BNT_dwa.npy')
                ds8_mat = np.load('./c_ells_BNT/dcls_BNT_ds8.npy')
                daia_mat = np.load('./c_ells_BNT/dcls_BNT_daia.npy')
                if self.kcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_kcut(cls_mat)
                                dom_mat = self.apply_kcut(dom_mat)
                                dob_mat = self.apply_kcut(dob_mat)
                                dh_mat = self.apply_kcut(dh_mat)
                                dns_mat = self.apply_kcut(dns_mat)
                                dw0_mat = self.apply_kcut(dw0_mat)
                                dwa_mat = self.apply_kcut(dwa_mat)
                                ds8_mat = self.apply_kcut(ds8_mat)
                                daia_mat = self.apply_kcut(daia_mat)
                elif self.lcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_lcut(cls_mat)
                                dom_mat = self.apply_lcut(dom_mat)
                                dob_mat = self.apply_lcut(dob_mat)
                                dh_mat = self.apply_lcut(dh_mat)
                                dns_mat = self.apply_lcut(dns_mat)
                                dw0_mat = self.apply_lcut(dw0_mat)
                                dwa_mat = self.apply_lcut(dwa_mat)
                                ds8_mat = self.apply_lcut(ds8_mat)
                                daia_mat = self.apply_lcut(daia_mat)

            else:
                raise Exception('Fisher computation cat must be def or BNT.')

            new_ls = self.ells

            full_fs_mat = np.matrix(np.zeros((8, 8), dtype=float))
            if self.lcut is None:
                rangepar = len(new_ls)
            else:
                rangepar = self.ind_cuts
            for l in range(rangepar):

                params_list = [np.matrix(dom_mat[l]), np.matrix(dob_mat[l]),
                               np.matrix(dw0_mat[l]), np.matrix(dwa_mat[l]),
                               np.matrix(dh_mat[l]), np.matrix(dns_mat[l]),
                               np.matrix(ds8_mat[l]), np.matrix(daia_mat[l])]

                cur_cls = np.matrix(cls_mat[l])
                inv_cs = self.inv(cur_cls)

                f_list = []
                for pa in range(len(params_list)):
                    for pb in range(len(params_list)):
                        if pb >= pa:
                            f_list.append(
                                self.in_fish_sum(l, params_list[pa],
                                                 params_list[pb], inv_cs))

                f_l_mat = np.matrix(np.zeros((8, 8), dtype=float))

                pl_list = [0, 8, 7, 6, 5, 4, 3, 2, 1]

                for t in range(8):
                    f_l_mat[t, t:] = np.array(f_list)[
                                     np.sum(pl_list[:t + 1]):np.sum(
                                         pl_list[:t + 2])]
                    f_l_mat[t + 1:, t] = np.array(f_list)[
                                         np.sum(pl_list[:t + 1]) + 1:np.sum(
                                             pl_list[:t + 2])].reshape(
                        7 - t, 1)

                full_fs_mat = copy.deepcopy(full_fs_mat) + f_l_mat
        elif self.survey == 'kinematic':
            f_sky = 5000 * (np.pi / 180) ** 2 / (4 * np.pi)

            if cat == 'def':
                cls_mat = np.load('./c_ells_default/cls_fiducial_kinematic.npy')
                dom_mat = np.load('./c_ells_default/dcls_kinematic_dom.npy')
                dob_mat = np.load('./c_ells_default/dcls_kinematic_dob.npy')
                dh_mat = np.load('./c_ells_default/dcls_kinematic_dh.npy')
                dns_mat = np.load('./c_ells_default/dcls_kinematic_dns.npy')
                dw0_mat = np.load('./c_ells_default/dcls_kinematic_dw0.npy')
                dwa_mat = np.load('./c_ells_default/dcls_kinematic_dwa.npy')
                ds8_mat = np.load('./c_ells_default/dcls_kinematic_ds8.npy')
            elif cat == 'BNT':
                cls_mat = np.load('./c_ells_BNT/cls_BNT_fiducial_kinematic.npy')
                dom_mat = np.load('./c_ells_BNT/dcls_BNT_kinematic_dom.npy')
                dob_mat = np.load('./c_ells_BNT/dcls_BNT_kinematic_dob.npy')
                dh_mat = np.load('./c_ells_BNT/dcls_BNT_kinematic_dh.npy')
                dns_mat = np.load('./c_ells_BNT/dcls_BNT_kinematic_dns.npy')
                dw0_mat = np.load('./c_ells_BNT/dcls_BNT_kinematic_dw0.npy')
                dwa_mat = np.load('./c_ells_BNT/dcls_BNT_kinematic_dwa.npy')
                ds8_mat = np.load('./c_ells_BNT/dcls_BNT_kinematic_ds8.npy')
                if self.kcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_kcut(cls_mat)
                                dom_mat = self.apply_kcut(dom_mat)
                                dob_mat = self.apply_kcut(dob_mat)
                                dh_mat = self.apply_kcut(dh_mat)
                                dns_mat = self.apply_kcut(dns_mat)
                                dw0_mat = self.apply_kcut(dw0_mat)
                                dwa_mat = self.apply_kcut(dwa_mat)
                                ds8_mat = self.apply_kcut(ds8_mat)
                elif self.lcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_lcut(cls_mat)
                                dom_mat = self.apply_lcut(dom_mat)
                                dob_mat = self.apply_lcut(dob_mat)
                                dh_mat = self.apply_lcut(dh_mat)
                                dns_mat = self.apply_lcut(dns_mat)
                                dw0_mat = self.apply_lcut(dw0_mat)
                                dwa_mat = self.apply_lcut(dwa_mat)
                                ds8_mat = self.apply_lcut(ds8_mat)

            else:
                raise Exception('Fisher computation cat must be def or BNT.')

            new_ls = self.ells

            full_fs_mat = np.matrix(np.zeros((7, 7), dtype=float))
            if self.lcut is None:
                rangepar = len(new_ls)
            else:
                rangepar = self.ind_cuts
            for l in range(rangepar):

                params_list = [np.matrix(dom_mat[l]), np.matrix(dob_mat[l]),
                               np.matrix(dw0_mat[l]), np.matrix(dwa_mat[l]),
                               np.matrix(dh_mat[l]), np.matrix(dns_mat[l]),
                               np.matrix(ds8_mat[l])]

                cur_cls = np.matrix(cls_mat[l])
                # inv_cs = cur_cls.I
                inv_cs = self.inv(cur_cls)

                f_list = []
                for pa in range(len(params_list)):
                    for pb in range(len(params_list)):
                        if pb >= pa:
                            f_list.append(
                                self.in_fish_sum(l, params_list[pa],
                                                 params_list[pb], inv_cs))

                f_l_mat = np.matrix(np.zeros((7,7), dtype=float))

                pl_list = [0,7,6,5,4,3,2,1]

                for t in range(7):
                    f_l_mat[t, t:] = np.array(f_list)[
                                     np.sum(pl_list[:t + 1]):np.sum(
                                         pl_list[:t + 2])]
                    f_l_mat[t + 1:, t] = np.array(f_list)[
                                         np.sum(pl_list[:t + 1]) + 1:np.sum(
                                             pl_list[:t + 2])].reshape(
                        6 - t, 1)

                full_fs_mat = copy.deepcopy(full_fs_mat) + f_l_mat

        return f_sky * full_fs_mat

    def error_comp(self, cat='def'):
        """
        Computes the cosmological parameter constraints for either the
        standard or BNT transformed C_\ells.

        Parameters
        ----------
        cat: str
            Category of constraints to compute. Either 'def' for the
            standard C_\ells, or 'BNT' for the BNT transformed C_\ells.
        Returns
        -------
        dict
            Dictionary of cosmological parameters with 1\sigma constraints.
        """
        fisherin = self.fisher(cat=cat)
        cov = np.linalg.inv(fisherin)
        zero_dict = {'Om': np.sqrt(abs(cov[0, 0])),
                     'Ob': np.sqrt(abs(cov[1, 1])),
                     'w0': np.sqrt(abs(cov[2, 2])),
                     'wa': np.sqrt(abs(cov[3, 3])),
                     'h': np.sqrt(abs(cov[4, 4])),
                     'ns': np.sqrt(abs(cov[5, 5])),
                     'sigma_8': np.sqrt(abs(cov[6, 6]))}
        print(zero_dict)
        return zero_dict

    def bias_calc(self, alph, cat='def', correction='RS'):
        """
        Computes the cosmological parameter bias from neglecting the reduced
        shear correction for either the standard or BNT transformed
        C_\ells.

        Parameters
        ----------
        alph: str
            Parameter for which to compute bias. Must be one from:
            'Om','Ob', 'Ode', 'w0', 'wa', 'h', 'ns', 'sigma_8.
        cat: str
            Category of biases to compute. Either 'def' for the
            standard C_\ells, or 'BNT' for the BNT transformed C_\ells.
        Returns
        -------
        float
            Value of bias.
        """
        if self.survey == 'kinematic' and correction != 'RS':
            raise Exception('# Only the reduced shear correction can be'
                            'computed for a kinematic survey.')
        if cat == 'BNT' and correction != 'RS':
            raise Exception('# k-cut BNT analysis is currently only supported'
                            'for the Reduced Shear correction.')
        if alph not in ['Om','Ob', 'w0', 'wa', 'h', 'ns', 'sigma_8']:
            raise Exception('Invalid Parameter to calculate biases for')
        if self.survey == 'photometric':
            f_sky = 15000 * (np.pi / 180) ** 2 / (4 * np.pi)
            if cat == 'def':
                if correction == 'RS':
                    file_name = 'del_cls_fiducial.npy'
                elif correction == 'Doppler':
                    file_name = 'del_cls_doppler.npy'
                elif correction == 'Doppler-IA':
                    file_name = 'del_cls_IA_doppler.npy'
                param_dict = {'Om': [0, './c_ells_default/dcls_default_dom.npy'],
                              'Ob': [1, './c_ells_default/dcls_default_dob.npy'],
                              'w0': [2, './c_ells_default/dcls_default_dw0.npy'],
                              'wa': [3, './c_ells_default/dcls_default_dwa.npy'],
                              'h': [4, './c_ells_default/dcls_default_dh.npy'],
                              'ns': [5, './c_ells_default/dcls_default_dns.npy'],
                              'sigma_8': [6, './c_ells_default/dcls_default_ds8.npy']}
                RS_corr_arr = np.load('./c_ells_default/' + file_name)
                cls_mat = np.load('./c_ells_default/cls_fiducial.npy')

            elif cat == 'BNT':
                param_dict = {'Om': [0, './c_ells_BNT/dcls_BNT_dom.npy'],
                              'Ob': [1, './c_ells_BNT/dcls_BNT_dob.npy'],
                              'w0': [2, './c_ells_BNT/dcls_BNT_dw0.npy'],
                              'wa': [3, './c_ells_BNT/dcls_BNT_dwa.npy'],
                              'h': [4, './c_ells_BNT/dcls_BNT_dh.npy'],
                              'ns': [5, './c_ells_BNT/dcls_BNT_dns.npy'],
                              'sigma_8': [6, './c_ells_BNT/dcls_BNT_ds8.npy']}
                RS_corr_arr = np.load('./c_ells_BNT/del_cls_BNT_fiducial.npy')
                cls_mat = np.load('./c_ells_BNT/cls_BNT_fiducial.npy')
                if self.kcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_kcut(cls_mat)
                                RS_corr_arr = self.apply_kcut(RS_corr_arr)
                elif self.lcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_lcut(cls_mat)
                                RS_corr_arr = self.apply_lcut(RS_corr_arr)

            else:
                raise Exception('Invalid category for biases, choose def or BNT.')
        elif self.survey == 'kinematic':
            f_sky = 5000 * (np.pi / 180) ** 2 / (4 * np.pi)
            if cat == 'def':
                param_dict = {
                    'Om': [0, './c_ells_default/dcls_kinematic_dom.npy'],
                    'Ob': [1, './c_ells_default/dcls_kinematic_dob.npy'],
                    'w0': [2, './c_ells_default/dcls_kinematic_dw0.npy'],
                    'wa': [3, './c_ells_default/dcls_kinematic_dwa.npy'],
                    'h': [4, './c_ells_default/dcls_kinematic_dh.npy'],
                    'ns': [5, './c_ells_default/dcls_kinematic_dns.npy'],
                    'sigma_8': [6, './c_ells_default/dcls_kinematic_ds8.npy']}
                RS_corr_arr = np.load('./c_ells_default/del_cls_kinematic.npy')
                cls_mat = np.load('./c_ells_default/cls_fiducial_kinematic.npy')

            elif cat == 'BNT':
                param_dict = {'Om': [0, './c_ells_BNT/dcls_BNT_kinematic_dom.npy'],
                              'Ob': [1, './c_ells_BNT/dcls_BNT_kinematic_dob.npy'],
                              'w0': [2, './c_ells_BNT/dcls_BNT_kinematic_dw0.npy'],
                              'wa': [3, './c_ells_BNT/dcls_BNT_kinematic_dwa.npy'],
                              'h': [4, './c_ells_BNT/dcls_BNT_kinematic_dh.npy'],
                              'ns': [5, './c_ells_BNT/dcls_BNT_kinematic_dns.npy'],
                              'sigma_8': [6, './c_ells_BNT/dcls_BNT_kinematic_ds8.npy']}
                RS_corr_arr = np.load('./c_ells_BNT/del_cls_BNT_kinematic.npy')
                cls_mat = np.load('./c_ells_BNT/cls_BNT_fiducial_kinematic.npy')
                if self.kcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_kcut(cls_mat)
                                RS_corr_arr = self.apply_kcut(RS_corr_arr)
                elif self.lcut is not None:
                    for i_ind in range(10):
                        for j_ind in range(10):
                            if j_ind >= i_ind:
                                cls_mat = self.apply_lcut(cls_mat)
                                RS_corr_arr = self.apply_lcut(RS_corr_arr)
            else:
                raise Exception(
                    'Invalid category for biases, choose def or BNT.')

        fish = self.fisher(cat)
        inv_fish = self.inv(fish)
        alpha = param_dict[alph][0]

        ells = self.ells

        b_list = []
        for bet in param_dict.keys():
            beta = param_dict[bet][0]

            d_div_mat = np.load(param_dict[bet][1])
            if self.kcut is not None:
                for i_ind in range(10):
                    for j_ind in range(10):
                        if j_ind >= i_ind:
                            d_div_mat = self.apply_kcut(d_div_mat)
            elif self.lcut is not None:
                for i_ind in range(10):
                    for j_ind in range(10):
                        if j_ind >= i_ind:
                            d_div_mat = self.apply_lcut(d_div_mat)

            l_list = []
            if self.lcut is None:
                rangepar = len(ells)
            else:
                rangepar = self.ind_cuts
            for mode in range(rangepar):
                if mode == 0:
                    del_l = 10.0
                else:
                    del_l = ells[mode] - ells[mode - 1]
                inv_c_mat = self.inv(np.matrix(cls_mat[mode]))

                dB_mat = np.matrix(d_div_mat[mode])
                dCl_mat = np.matrix(RS_corr_arr[mode])

                f_val_l = inv_c_mat * dCl_mat * inv_c_mat * dB_mat
                this_l = (ells[mode] + 0.5) * del_l * f_val_l.trace()[0, 0]
                l_list.append(this_l)

            b_val = f_sky * inv_fish[alpha, beta] * np.sum(l_list)
            b_list.append(b_val)
        reval = np.sum(b_list)
        return reval


if __name__ == '__main__':
    print('# Running kcut_rs.py procedurally, computing Stage IV optimal '
          'k-cut constraints and biases.')
    fishinst = Fishercalc(survey='photometric', kcut=3.6)
    zero_dict = fishinst.error_comp(cat='BNT')
    print('Param  \sigma                Bias                    Bias/\sigma')
    for i in range(len(zero_dict.keys())):
        print(list(zero_dict.keys())[i], zero_dict[list(zero_dict.keys())[i]],
              fishinst.bias_calc(list(zero_dict.keys())[i], 'BNT'),
              fishinst.bias_calc(list(zero_dict.keys())[i], 'BNT') /
              list(zero_dict.values())[i])
