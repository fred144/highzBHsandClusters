import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from astropy.io import fits
from scipy.stats import norm
import glob
from scipy import interpolate
from scipy import integrate
from scipy.integrate import solve_ivp
from astropy import cosmology
import h5py
from scipy.interpolate import RegularGridInterpolator
class CoolingFunctionInterpolator:
    def __init__(self, file_path):
        self.file_path = file_path
        self._load_data()
        self.interpolator = RegularGridInterpolator(
            (self.temperatures, self.metallicities),
            self.lambda_values,
            bounds_error=False,
            fill_value=None,
        )

    def _load_data(self):
        # Load the data using numpy, skipping header lines
        data = np.loadtxt(self.file_path, skiprows=5)
        self.temperatures = data[:, 0]
        self.lambda_values = data[:, 1:]

        # Extract metallicities from the header line directly
        with open(self.file_path, "r") as file:
            metallicity_line = file.readlines()[3].strip().split()
            # print(metallicity_line)
            self.metallicities = list(
                map(float, metallicity_line[1 : len(self.lambda_values[0]) + 1])
            )
            # print(self.metallicities)

    def cooling_function(self, temperature, metallicity):
        log_temp = np.log10(temperature)
        log_metallicity = np.log10(metallicity)
        # print(log_temp,  log_metallicity)
        log_lambda = self.interpolator((log_temp, log_metallicity))
        return 10**log_lambda

    def custom_cooling_function(
        self, temperature, metallicity, T_thresh_cool=1e5, T_slope=0.1
    ):
        """
        a custom cooling function that after T_thresh_cool, the cooling function
        grown as T^0.1, still shifts up with increasing metallicity
        """

        # ensure arrays for vectorized operations
        temps = np.asarray(temperature)
        Z = np.asarray(metallicity)

        # avoid non-positive metallicities (log10 in interpolator)
        if np.any(Z <= 0):
            Z = np.where(Z <= 0, 1e-4, Z)

        # remember if inputs were scalars so we can return scalars
        temps_was_scalar = np.isscalar(temperature) or temps.shape == ()
        Z_was_scalar = np.isscalar(metallicity) or Z.shape == ()

        # lambda_base: cooling at the requested temperatures (array or scalar)
        lambda_base = self.cooling_function(temps, Z)

        # cooling at the threshold for the given metallicity(s)
        lambda_thresh = self.cooling_function(T_thresh_cool, Z)

        # compute power-law growth above threshold
        # broadcast temps and lambda_thresh as necessary
        # use where to switch between base and extrapolated regimes
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = (temps / T_thresh_cool) ** T_slope
        lambda_custom = np.where(
            temps > T_thresh_cool, lambda_thresh * factor, lambda_base
        )

        # return scalar if inputs were scalar
        if temps_was_scalar and Z_was_scalar:
            return np.asarray(lambda_custom).item()
        return lambda_custom


def cooling_fn_generator(path):
    """returns cooling fn generator you can call based on path,
    if path does not exist, it will generate the cooling function from hdf5 files

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    file = glob.glob(path)
    if len(file) > 0:
        print("*** reading file", file)
        data = np.load(file[0])
        Lambda_tab = data["Lambda_tab"]
        redshifts = data["redshifts"]
        Zs = data["Zs"]
        log_Tbins = data["log_Tbins"]
        log_nHbins = data["log_nHbins"]
        Lambda = interpolate.RegularGridInterpolator(
            (log_nHbins, log_Tbins, Zs, redshifts),
            Lambda_tab,
            bounds_error=False,
            fill_value=1e-30,
        )
    else:
        files = np.sort(glob.glob("z_*hdf5"))  ## CHANGE FOR WHERE YOU WANT TO KEEP IT
        redshifts = np.array([float(f[-10:-5]) for f in files])
        HHeCooling = {}
        ZCooling = {}
        TE_T_n = {}
        for i in range(len(files)):
            f = h5py.File(files[i], "r")
            i_X_He = -3
            Metal_free = f.get("Metal_free")
            Total_Metals = f.get("Total_Metals")
            log_Tbins = np.array(np.log10(Metal_free["Temperature_bins"]))
            log_nHbins = np.array(np.log10(Metal_free["Hydrogen_density_bins"]))
            Cooling_Metal_free = np.array(Metal_free["Net_Cooling"])[
                i_X_He
            ]  ##### what Helium_mass_fraction to use    Total_Metals = f.get('Total_Metals')
            Cooling_Total_Metals = np.array(Total_Metals["Net_cooling"])
            HHeCooling[redshifts[i]] = interpolate.RectBivariateSpline(
                log_Tbins, log_nHbins, Cooling_Metal_free
            )
            ZCooling[redshifts[i]] = interpolate.RectBivariateSpline(
                log_Tbins, log_nHbins, Cooling_Total_Metals
            )
            f.close()
        Lambda_tab = np.array(
            [
                [
                    [
                        [
                            HHeCooling[zz].ev(lT, ln) + Z * ZCooling[zz].ev(lT, ln)
                            for zz in redshifts
                        ]
                        for Z in Zs
                    ]
                    for lT in log_Tbins
                ]
                for ln in log_nHbins
            ]
        )
        np.savez(
            "./data/Lambda_tab_redshifts.npz",
            Lambda_tab=Lambda_tab,
            redshifts=redshifts,
            Zs=Zs,
            log_Tbins=log_Tbins,
            log_nHbins=log_nHbins,
        )
        Lambda = interpolate.RegularGridInterpolator(
            (log_nHbins, log_Tbins, Zs, redshifts),
            Lambda_tab,
            bounds_error=False,
            fill_value=0,
        )
        print("interpolated lambda")
    return Lambda

#%% explore the cooling function
# cooling_fn = cooling_fn_generator("../tables/Lambda_tab_redshifts.npz")



# def mass_evolution(t, mass):
#     """Ode to solve the mass evolution of the galaxy

#     Args:
#         t (_type_): _description_
#         mass (_type_): _description_

#     Returns:
#         _type_: _description_
#     """

#     #  t: time (units: Gyr)
#     #  mass [0-4] (units: solar mass): 5 vector of mass of each component/term
#     #  energy [5-9] (units: erg): 5 vector of energy of each component/term

#     # get current redshift
#     z = cosmology.z_at_value(LCDM.age, t * u.Gyr)

#     m_gas = mass[0] * u.solMass
#     m_star = mass[1] * u.solMass
#     m_cgm = mass[2] * u.solMass  # Total CGM mass
#     m_metals = mass[3] * u.solMass
#     m_halo = mass[4] * u.solMass  # total halo mass

#     e_ism_wind = mass[5] * u.erg  # Energy gained from energy-loaded galactic winds
#     e_cgm_cool = mass[6] * u.erg  # Energy loss from gas precipitation onto the Galaxy
#     e_cgm_out = mass[7] * u.erg  # Energy loss from mass ejected from the CGM into IGM
#     e_cgm_in = mass[8] * u.erg  # Energy gained from mass accretion from the IGM
#     e_cgm = mass[9] * u.erg  # Total CGM energy

#     # now, derive relevant quantities
#     t_depletion = depletion_time(z, m_star, exp, dep_time_norm)
#     halo_rvir = virial_radius(z, m_halo).to(u.kpc)
#     halo_vir_temp = virial_T(m_halo, halo_rvir).to(u.K)
#     t_dynamical = t_ff(halo_rvir, halo_rvir, m_halo).to(u.Gyr)
#     rho_crit = LCDM.critical_density(z)
#     cgm_metallicity = m_metals / m_cgm
#     cgm_metallicity_sol = cgm_metallicity / Z_sol

#     # can also use   # t_dep = depletion_time_McGaugh(z,mstar.value) * u.Gyr

#     # cooling function value at this timestep
#     cooling_lambda = cooling_fn(
#         (-4, np.log10(halo_vir_temp.value), cgm_metallicity_sol, 0)
#     ) * (u.erg * u.cm**3 * u.s**-1)

#     # compute density normalization for power-law density model from CGM mass
#     r1 = 0.1 * halo_rvir  # our definition of inner radius of CGM

#     ## Another possible definition of inner radius of CGM
#     # r1 = 10.0*u.kpc

#     # get the central density in units of kpc^-3 ?
#     rho0 = density0(mCGM=m_cgm, r1=r1, Rvir=halo_rvir, alpha=alpha)

#     # estimate energy ejection loss timescale and limitit to dynamical time
#     c_sound = np.sqrt(e_cgm / m_cgm)  # approximate sound speed
#     t_ejection = halo_rvir / c_sound  # ejection time of hot gas
#     t_ejection = min(max(t_ejection, 0.1 * t_dynamical), t_dynamical)
#     # print(t_ejection)
#     # outflow rate assuming that gas flows out at the sound speed
#     # eq. 18 from Carr+ 2023
#     # Energy ejection loss rate = (E_CGM - E_vir)/t_ej
#     dot_e_cgm_out = (
#         max(e_cgm - kb * halo_vir_temp * m_cgm / mu, 0.0 * u.erg) / t_ejection
#     ).to(u.erg / u.Gyr)

#     # radiative losses in the cgm, integrated, Eq. 3 from Carr 2023
#     dot_e_cgm_cool = energy_loss(
#         Lamb=cooling_lambda, Rvir=halo_rvir, r1=r1, rho0=rho0
#     ).to(u.erg / u.Gyr)

#     # speceific energy of eject gas
#     cgm_specific_e = cgm_ejecti_specific_energy_ratio * max(
#         e_cgm / m_cgm, kb * halo_vir_temp / mu
#     )

#     # (effective) cooling time of CGM
#     tcool_eff = (cgm_specific_e / dot_e_cgm_cool) * m_cgm
#     #   tcomp = 1.2e7*u.yr * ((1+20)/(1+z))**4
#     #   tcool_eff = min(tcool_eff, tcomp)  # include Compton cooling
#     tcool_eff = tcool_eff + 1 * t_dynamical  # include dynamical time?
#     dot_e_cgm_cool = (e_cgm / tcool_eff).to(u.erg / u.Gyr)

#     ##################### moving on to mass evolution
#     # CGM_eject_dot = max(mCGM - mCGM_precip, 0 * u.solMass) / (t_dyn)
#     # CGM_eject_dot = (energy_gain(eta_e, mstar_dot) * (mu/(kb*Tvir))).to(u.solMass/u.Gyr)

#     # CGM mass loss due to cooling, eq. 7 from Carr 2023
#     dot_m_cgm_cool = (m_cgm / tcool_eff).to(u.solMass / u.Gyr)
#     # eq. 12, m_gas is kinda the ism
#     dot_m_sfr = m_gas / t_depletion
#     # eq 11 ish
#     dot_m_gas = (dot_m_cgm_cool - dot_m_sfr * (1 + eta_m)).to(u.solMass / u.Gyr)
#     dot_m_sfr = dot_m_sfr.to(u.solMass / u.Gyr)

#     dot_m_halo = halo_infall(z, m_halo)
#     dot_m_cgm_in = fb * dot_m_halo  # eq. 6

#     # CGM eject loss term, eq 10
#     dot_m_cgm_out = ((1 / cgm_specific_e) * dot_e_cgm_out).to(u.solMass / u.Gyr)

#     ################## calculate the energies that depend on the mass changes

#     # ratio of dot ECGM_in / ECGM_out
#     e_ejection_to_infall_ratio = cgm_infall_prevention_const / (
#         dot_e_cgm_out / ((kb * halo_vir_temp / mu) * dot_m_cgm_in)
#     ).to(u.dimensionless_unscaled)
#     # equation 19 CGM infall prevention factor
#     f_prevent = min(max(e_ejection_to_infall_ratio, 0.1), 1.0)  # 0.1 < f < 1
#     dot_m_cgm_in *= f_prevent

#     # energy input from SF
#     dot_e_ism_wind = energy_gain(eta_e, dot_m_sfr)

#     # energy due to accretion, eq 16
#     dot_e_cgm_in = (kb * halo_vir_temp / mu * dot_m_cgm_in).to(u.erg * u.Gyr**-1)

#     # CGM feedback gain term, eq 9
#     dot_m_ism_wind = dot_m_sfr * eta_m

#     ##################  total Mass, Energy, and Metallicity derivatives
#     dot_m_cgm = dot_m_cgm_in + dot_m_ism_wind - dot_m_cgm_cool - dot_m_cgm_out
#     dot_m_metal = (
#         metal_yield * eta_z * dot_m_sfr
#         + Z_IGM * Z_sol * dot_m_cgm_in
#         - cgm_metallicity * (dot_m_cgm_cool + dot_m_cgm_out)
#     )
#     dot_e_cgm = dot_e_ism_wind + 1 * dot_e_cgm_in - dot_e_cgm_out - 1 * dot_e_cgm_cool

#     # limiter
#     if (m_cgm.value < 3e5) & (dot_m_cgm < 0):
#         dot_m_cgm *= max((m_cgm.value - 5e3) / 5e3, 0)

#     # t_dep_list.append(t_dep.value)

#     derivs = np.array(
#         [
#             dot_m_gas.value,
#             dot_m_sfr.value,
#             dot_m_cgm.value,
#             dot_m_metal.value,
#             dot_m_halo.value,
#             dot_e_ism_wind.value,
#             dot_e_cgm_cool.value,
#             dot_e_cgm_out.value,
#             dot_e_cgm_in.value,
#             dot_e_cgm.value,
#         ]
#     )
#     return derivs