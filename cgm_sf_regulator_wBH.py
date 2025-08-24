# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp
from astropy import cosmology
import scipy
import cmasher as cmr
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# import seaborn as sns
from regulator_lib.cooling_fn_generator import cooling_fn_generator
import astropy.constants as consts
import astropy.units as u

from tqdm import tqdm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import RegularGridInterpolator

# standard flat cosmology
H0 = 70
h = 0.7
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
f_baryon = Ob0 / Omegam0  # universal baryon fraction
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": False,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)


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


# interpolator = CoolingFunctionInterpolator("./tables/newcool_viraj.dat")
# lambda_value = interpolator.cooling_function(1e4, 0.7)


def custom_mass_loading(mhalo, A=10, alpha=-1.4):
    """mass loading factor as a function of halo mass"""
    return A * (mhalo / (1e10 * u.solMass)) ** alpha


def custom_energy_loading(mhalo_z0, A=0.10, alpha=-0.5):
    """energy loading fact or as a function of halo mass"""
    eta_e = A * (mhalo_z0 / (1e12 * u.solMass)) ** alpha
    if np.any(eta_e > 1):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


def vcirc_energy_loading(halo_vcirc, alpha_e=0.1):
    eta_e = alpha_e * (halo_vcirc.value / 200) ** (-3 / 2)

    # if eta e > 1 set to 1, halo_vcirc can be float or array
    if np.any(eta_e > 1):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


def vcirc_mass_loading(halo_vcirc, alpha_m=9):
    return alpha_m * (halo_vcirc.value / 200) ** (-3 / 2)


def t_ff(r, Rvir, mhalo):
    """free-fall time as a function of radius

    Args:
        r (_type_): radius
        Rvir (_type_): virial radius
        mhalo (_type_): halo mass

    Returns:
        _type_: _description_
    """
    # Menc = mhalo*(r/Rvir)**3
    G = consts.G
    v_circ = np.sqrt(G * mhalo / Rvir) * 1.3
    return r / v_circ


def depletion_time(z, mstar, exp, dep_time_norm):
    """depletion time (power-law),


    Args:
        z (_type_): redshift
        mstar (_type_): mass of galaxy in solar mass
        exp (_type_): _description_
        dep_time_norm (_type_): _description_

    Returns:
        _type_: _description_
    """
    tH = (1 / LCDM.H(z=z)).to(u.Gyr)
    return dep_time_norm * tH * (mstar / (4e10 * u.solMass)) ** (-exp)


def depletion_time_test(m_ism_total, r_ism):

    vcirv = np.sqrt(consts.G.to("kpc**3 / (Msun s**2)") * m_ism_total / r_ism)
    t_dyn = r_ism / (vcirv * 0.5)
    t_dyn = t_dyn.to(u.Gyr)
    return t_dyn


def depletion_time_McGaugh(z, mstar):
    """depletion time fit from McGaugh observations
     eq. 13 from Carr 2023

    Args:
        z (_type_): _description_
        mstar (_type_): _description_

    Returns:
        _type_: _description_
    """
    tH = (1 / LCDM.H(z=z)).to(u.Gyr)
    tH0 = (1 / LCDM.H(z=0)).to(u.Gyr)
    if mstar < 5e7:
        tdep = 10 ** (4.92 - 0.37 * np.log10(5e7)) * (tH / tH0)
    else:
        tdep = 10 ** (4.92 - 0.37 * np.log10(mstar)) * (tH / tH0)
    return tdep


def virial_radius(z, mhalo, Delc=200):
    """Halo virial radius, classical 200 topahat overdensity
    Args:
        z (_type_): _description_
        mhalo (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (mhalo / (LCDM.critical_density(z) * (4 / 3) * np.pi * Delc)) ** (1 / 3)


def halo_infall_dekel(z, mhalo):
    """halo mass inflows, from # d M_{halo} / dt (Dekel et al 2009)
    also, Carr 2023 Eq 4

    Args:
        z (_type_): _description_
        mhalo (_type_): _description_

    Returns:
        _type_: \dot{M}_{halo}
    """
    mdot = (
        0.47
        * mhalo
        * (mhalo / (1e12 * u.solMass)) ** (0.15)
        * ((1 + z) / 3) ** (2.25)
        * u.Gyr ** (-1)
    ).to(u.solMass / u.Gyr)
    try:
        if mdot <= 0:
            print("____", mhalo)
    except:
        pass

    return mdot


def halo_infall_fakhouri(z, mhalo):
    """Fakhouri + 2011

    Args:
        z (_type_): _description_
        mhalo (_type_): _description_
    """
    mean_mdot = (
        46.1
        * (u.solMass / u.yr)
        * (mhalo / (1e12 * u.solMass)) ** 1.1
        * (1.0 + 1.11 * z)
        * ((Omegam0 * (1 + z) ** 3) + Omegade0) ** 0.5
    )
    mean_mdot = mean_mdot.to(u.solMass / u.Gyr)
    return mean_mdot


def virial_T(mhalo, Rvir):
    """Halo fp

    Args:
        mhalo (_type_): halo mass
        Rvir (_type_): halo virial radius

    Returns:
        _type_: virial temperture
    """
    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return (2 / 5) * ((G * mhalo * mp) / (Rvir * kb))


def density0(mCGM, r0, Rvir, alpha=1.4):
    """assuming rho = rho0 * (r/r1)^-alpha, this function computes rho0

    Args:
        mCGM (_type_): total mass of the CGM
        alpha (_type_): some power-law index
        r1 (_type_): _description_
        Rvir (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (mCGM * (3 - alpha)) / (
        (4 * np.pi * r0 ** (3) * ((Rvir / r0) ** (3 - alpha) - 1))
    )


def halo_mass_growth(t, mass):
    """
    # Halo mass evolution called by initial_mhalo to
    # estimate initial halo mass at arbitrary redshift
    """
    mhalo = mass * u.solMass
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift
    if t > 13.47:  # 13.4 is the age of the universe in Gyr
        t = 13.466983947061877
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    mhalo_dot = halo_infall_fakhouri(z, mhalo)

    return mhalo_dot


def mhalo_at_z0(mhalo_at_z, z_obs):
    """Finds the halo mass at z=0.0001 given its mass at a specified redshift.

    Args:
        mhalo_at_z (_type_): Halo mass at the observed redshift.
        z_obs (_type_): Observed redshift.

    Returns:
        _type_: Halo mass at z=0.0001.
    """
    time_interval = (LCDM.age(z_obs).value, LCDM.age(0.0001).value)
    # print(time_interval)
    mass_initial = np.array([mhalo_at_z.value])
    sol_0 = solve_ivp(halo_mass_growth, time_interval, mass_initial)
    return sol_0.y[0][-1]  # return the last value of the solution


def energy_gain(eta_e, mstar_dot):
    """energy associated with SN-powered galactic winds,
    Carr 2023 Eq 17

    Args:
        eta_e (_type_): _description_
        mstar_dot (_type_): _description_

    Returns:
        _type_: _description_
    """
    norm = 100  # per 100 solar masses
    e_perSNE = 1e51 * u.erg
    return (eta_e * (mstar_dot) * (e_perSNE / (norm * u.solMass))).to(u.erg * u.Gyr**-1)


def energy_loss(Lamb, Rvir, r1, rho0, mu, alpha):
    """radiative energy loss from the CGM
    Equation 3 from Carr 2023

    Args:
        Lamb (_type_): cooling function
        Rvir (_type_): virial radius
        r1 (_type_): characteristic radius
        rho0 (_type_): central density of the CGM

    Returns:
        _type_: _description_
    """
    rad_loss = (
        (4 * np.pi * ((rho0 / mu) ** 2) * (r1**3))
        * (Lamb)
        * (((Rvir / r1) ** (3 - 2 * alpha) - 1) / (3 - 2 * alpha))
    )
    if rad_loss < 0:
        print("Negative radiative", Lamb, rho0, mu, r1, Rvir, alpha)
    return rad_loss


def halo_mass_evol(t, mass):
    """
    # Halo mass evolution called by
    # initial_mhalo to estimate initial halo mass at arbitrary redshift
    """
    mhalo = mass * u.solMass
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    mhalo_dot = halo_infall_fakhouri(z, mhalo)
    return -mhalo_dot


def initial_mhalo(mhalo_z0, time_interval):
    """finds initial halo mass at the left edge of the time interval

    Args:
        mhalo_z0 (_type_): _description_
        time_interval (_type_): _description_

    Returns:
        _type_: _description_
    """

    mass_initial = np.array([mhalo_z0.value])
    time_at_z0 = LCDM.age(0.001).value
    time_interval = (time_interval[0], time_at_z0)
    sol_0 = solve_ivp(halo_mass_evol, time_interval, mass_initial)
    return sol_0.y[0][-1]  # return the last value of the solution


def rho_gas_0_term_integrand(x, b):
    return x**2 * (1 + x) ** (27 * b / (2 * x))


def rho_gas_0(gas_frac, rho_crit, delta_c, b, c):
    prefactor = gas_frac * (Ob0 / Omegam0) * rho_crit * delta_c * np.exp(27 * b / 2)
    prefactor *= np.log(1 + c) - (c / (1 + c))
    integral_result, err = scipy.integrate.quad(
        rho_gas_0_term_integrand, 0, c, args=(b)
    )
    # print("integral_result", integral_result)
    # print("b", b)
    # print("prefactor", prefactor)
    return prefactor * integral_result**-1


def rho_gas_makino(r, Mgas, Mbaryon, Mvir, z_red, halo_temp, Delc=200):
    """gas denisity profile of the CGM mkino et al 2020

    Args:
        r (_type_): _description_
        Mvir (_type_): _description_
        z_red (_type_): _description_
        halo_temp (_type_): _description_
        Delc (int, optional): _description_. Defaults to 200.

    Returns:
        _type_: _description_
    """
    c = 6
    mu = 0.59
    rho_crit_z = LCDM.critical_density(z_red)
    gamma = 1.5
    f_gas = Mgas / Mbaryon  # gas fraction of all baryons

    r_vir = virial_radius(z_red, Mvir, Delc).to(u.kpc)
    r_scale = r_vir / c
    t_eff = halo_temp * u.K
    delta_c = (Delc / 3) * (c**3 / (np.log(1 + c) - c / (1 + c)))
    # delta_c = 3e3* Omegam0 * (1 + z_red)**3
    b_of_m = (
        8
        * np.pi
        * consts.G
        * mu
        * consts.m_p
        * delta_c
        * rho_crit_z.to(u.kg * u.m**-3)
        * r_scale.to(u.m) ** 2
    )

    b_of_m /= (27 * consts.k_B * t_eff).to(u.m**2 * u.s**-2 * u.kg)
    # print(b_of_m)

    # if b is outside [1/3,5/3], pin
    if b_of_m < 1 / 3:
        b_of_m = 1 / 3
    elif b_of_m > 5 / 3:
        b_of_m = 5 / 3

    # print("b-values",  b_of_m)
    # print("Teff", t_eff)

    # b_of_r = (
    #     (2 / (9 * gamma))
    #     * (r / r_scale)
    #     * (np.log(1 + r / r_scale) - (r / (r + r_scale))) ** (-1)
    # )

    rho_cgm_0 = rho_gas_0(f_gas, rho_crit_z, delta_c, b_of_m, c)

    exponent = (27 * b_of_m) / (2 * r / r_scale)
    rho_gas = rho_cgm_0 * np.exp(-27 * b_of_m / 2) * (1 + r / r_scale) ** exponent

    return rho_gas


def rho_gas_makino_cm(r, Mgas, Mbaryon, Mvir, z_red, halo_temp, Delc=200):
    """gas denisity profile of the CGM mkino et al 2020, dimensionless, g/cm^3"""

    c = 6
    mu = 0.59
    rho_crit_z = LCDM.critical_density(z_red)
    gamma = 1.5
    f_gas = Mgas / Mbaryon  # gas fraction of all baryons

    r_vir = virial_radius(z_red, Mvir, Delc).to(u.cm)
    r_scale = r_vir / c
    t_eff = halo_temp * u.K
    delta_c = (Delc / 3) * (c**3 / (np.log(1 + c) - c / (1 + c)))
    # delta_c = 3e3* Omegam0 * (1 + z_red)**3

    b_of_m = (
        8
        * np.pi
        * consts.G.value
        * mu
        * consts.m_p.value
        * delta_c
        * rho_crit_z.to(u.kg * u.m**-3).value
        * r_scale.to(u.m).value ** 2
    )
    r_scale = r_scale.value  # now turn it into a number after the conversion

    b_of_m /= (27 * consts.k_B * t_eff).to(u.m**2 * u.s**-2 * u.kg).value
    # print(b_of_m)

    # if b is outside [1/3,5/3], pin
    if b_of_m < 1 / 3:
        b_of_m = 1 / 3
    elif b_of_m > 5 / 3:
        b_of_m = 5 / 3

    # print("b-values",  b_of_m)
    # print("Teff", t_eff)

    b_of_r = (
        (2 / (9 * gamma))
        * (r / r_scale)
        * (np.log(1 + r / r_scale) - (r / (r + r_scale))) ** (-1)
    )

    f_baryon = Ob0 / Omegam0
    rho_cgm_0 = rho_gas_0(f_gas, rho_crit_z, delta_c, b_of_m, c)

    rho_cgm_0 = rho_cgm_0.value

    exponent = (27 * b_of_m) / (2 * r / r_scale)
    rho_gas = rho_cgm_0 * np.exp(-27 * b_of_m / 2) * (1 + r / r_scale) ** exponent
    # print("---", rho_cgm_0)
    return rho_gas


def makino_cooling_intergrand(
    r, Mgas, Mbaryon, Mvir, z_red, halo_temp, cooling_lambda, Delc=200
):
    mp = consts.m_p.to(u.g)
    mu = 0.6

    profile = rho_gas_makino_cm(
        r, Mgas, Mbaryon, Mvir, z_red, halo_temp, Delc
    )  # in units of g/cm^3

    number_density = profile / (mu * mp)  # in units of cm^-3
    # print("***", number_density)
    # print(cooling_lambda.unit)
    integ = number_density.value**2 * cooling_lambda.value * 4 * np.pi * r**2
    # print(integ.unit)
    return integ  # in units of erg / (cm s)


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def exponential_disk(r, r_trunc_disk, rho_disk_central):
    """
    simple exponential disk profile
    r_trunc_disk is usually 0.02 of rvir

    """
    return rho_disk_central * np.exp(-r / r_trunc_disk)


def disk_central_density(m_total, r_trunc):
    """
    for stellar disk or ISM disk?
    """
    Sigma0 = m_total / (2 * np.pi * r_trunc**2)
    return Sigma0


def sfr_density_prof(r, r_trunc, Sigma0_gas, Sigma0_star, n=1.5):
    """
    units of msun/gyr/kpc^2
    is the integrand to solve for the total SFR
    """
    # solve for Sigma_gas, Sigma_star
    Sigma_gas = exponential_disk(r, r_trunc, Sigma0_gas)
    Sigma_star = exponential_disk(r, r_trunc, Sigma0_star)
    norm = 10  # *u.Msun / u.Gyr / u.kpc**2
    sfr_prof = norm * (Sigma_gas / Sigma_star) ** n
    return sfr_prof


# # #now, integrate the sfr_density from 0 to inf
# # Rvir = 10
# # r  = np.linspace(0, 100, 10) #* u.kpc

# # mstar = 1e3 #* u.Msun
# # m_gas = 1e3# * u.Msun

# # r_trunc = 0.02 * Rvir
# # Sigma0_star = disk_central_density(mstar,r_trunc )
# # Sigma0_gas = disk_central_density(m_gas,r_trunc )

# # Sigma_gas = exponential_disk(r, r_trunc , Sigma0_gas)
# # Sigma_star = exponential_disk(r,r_trunc,  Sigma0_star)

# # total_sfr, err = scipy.integrate.quad(sfr_density_prof, 0 ,Rvir, args=(r_trunc, Sigma0_gas, Sigma0_star))
# # print("total sfr", total_sfr)


#######
# fig,ax = plt.subplots(figsize=(8, 6), dpi=300)
# # Plot halo infall rates for Dekel and Fakhouri as a function of redshift
# halo_masses = np.geomspace(1e10, 1e12, 5) * u.solMass
# redshifts = np.linspace(0, 15, 100)

# fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# colors = sns.color_palette("viridis", len(halo_masses))

# for i, mhalo in enumerate(halo_masses):
#     dekel_infall_rates = [halo_infall_dekel(z, mhalo).value for z in redshifts]
#     fakhouri_infall_rates = [halo_infall_fakhouri(z, mhalo).value for z in redshifts]

#     ax.plot(redshifts, dekel_infall_rates, color=colors[i], label=f'Dekel, $M_{{halo}} = {mhalo.value:.1e} M_{{\odot}}$')
#     ax.plot(redshifts, fakhouri_infall_rates, color=colors[i], linestyle='--', label=f'Fakhouri, $M_{{halo}} = {mhalo.value:.1e} M_{{\odot}}$')

# ax.set_yscale('log')
# ax.set_xlabel('Redshift')
# ax.set_xlim(0, 15)
# ax.set_ylabel(r'$\dot{M}_{\rm halo}$ [$M_{\odot} \, \mathrm{Gyr}^{-1}$]')
# ax.set_title('Halo Infall Rates as a Function of Redshift')
# ax.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
# plt.show()

# # plot the CGM density profile
# r = np.linspace(0.1, 1, 100)
# mCGM = 1e3
# r1 = 0.1
# Rvir = 1
# alpha = 1.4
# rho0 = density0(mCGM, r1, Rvir, alpha)
# fig, ax = plt.subplots(1,1 , figsize=(5, 5), dpi=300)
# ax.plot(r, rho0 * (r / r1) ** (-alpha), label="alpha = {}".format(alpha))


class CGM_regulator:
    def __init__(
        self,
        mhalo_z0,
        time_interval,
        tstep=1,
        eta_z=0.3,
        verbose=False,
        dep_time_norm=0.4,
        cooling_dynamic_time_norm=1,
        disk_scale_length=0.02,
        KS_n=1.5,
        KS_kappa_s=1,
        r_bulge=1,
        ep_bh_radeff=0.1,
        ep_bh_feedback_eff=0.02,
        bondi_boost=100,
        dbug_norm_for_accretion_energy_rate=1
    ):
        self.mhalo_z0 = mhalo_z0
        self.verbose = verbose
        # self.mhalo_init = mhalo_init

        self.time_interval = time_interval
     
        self.eta_z = eta_z
        self.tstep = tstep
        self.evaluation_time_array = np.arange(
            self.time_interval[0], self.time_interval[1], self.tstep
        )

        # here we set some of the model params that are not the key parameters
        self.metal_yield = 0.02

        self.fb = 0.16  # universal baryon fraction
        self.Delc = 200  # overdensity
        self.G = consts.G  # Gravitational Constant
        self.kb = consts.k_B  # Boltzmann constant
        self.mp = consts.m_p  # mass of a proton (Hydrogen mass)
        self.Z_sol = 0.0134  # solar metallicty
        self.Z_IGM = 0.01  # Assumed metallicity of IGM in solar units
        self.mu = 0.6 * self.mp  # mean molecular weight
        self.alpha = 1.4  # density power-law index
        self.exp = 0.0  # T_depletion power-law
        self.dep_time_norm = dep_time_norm  # depletion time prefactor
        # cooling time prefactor for accretion
        self.grav_cooling_time_norm = 1
        # cooling time prefactor for the cooling of CGM
        self.cooling_time_norm = cooling_dynamic_time_norm

        self.t_eject_lim_norm = 0.1  # ejection time limit factor

        # sets how E_out is at preventing inflow (smaller - more effective)
        self.cgm_ejecti_specific_energy_ratio = 1

        # ratio of specific energy of ejected gas to CGM gas
        self.cgm_infall_prevention_const = 1

        self.disk_scale = disk_scale_length  # disk scale radius in units of Rvir
        self.r_cgm_scale = 0.1  # inner radius of CGM in units of Rvir

        # KS law star formation
        self.ks_n = KS_n
        self.ks_kappa_s = KS_kappa_s
        self.r_bulge = r_bulge  # kpc

        # self.cooling_fn = cooling_fn_generator(
        #     "./tables/Lambda_tab_redshifts.npz"
        # )  # resides outside the class
        
        # BH params
        self.ep_bh_radeff = ep_bh_radeff
        self.ep_bh_feedback_eff = ep_bh_feedback_eff  
        self.bondi_boost = bondi_boost
        
        self.cooling_fn = CoolingFunctionInterpolator("./tables/newcool_viraj.dat")

        self.zinit = cosmology.z_at_value(LCDM.age, time_interval[0] * u.Gyr)
        self.zfinal = cosmology.z_at_value(LCDM.age, time_interval[1] * u.Gyr)
        print("_________________________________________")
        print(
            "Running models from z = {:.2f} to z = {:.2f}".format(
                self.zinit, self.zfinal
            )
        )
        # print("Halo mass at z=0 = {:.2e}  Msol".format(mhalo_z0))

        # also declare the actual results to be stored
        self.ode_results = {}
        
        # dbug 
        self.dbug_norm_for_accretion_energy_rate = dbug_norm_for_accretion_energy_rate

    def mass_evolution(self, t, mass, ode_mode=True):
        """Ode to solve the mass evolution of the galaxy

        #  t: time (units: Gyr)
        #  mass [0-4] (units: solar mass): 5 vector of mass of each component/term
        #  energy [5-9] (units: erg): 5 vector of energy of each component/term

        """

        # get current redshift
        z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
        # fmt: off
        m_ism = mass[0] * u.solMass         # Total ISM mass
        m_star = mass[1] * u.solMass        # Total stellar mass
        m_bulge = mass[2] * u.solMass       # Total bulge mass
        # m_cgm = mass[2] * u.solMass       # Total CGM mass
        m_cgm_hot = mass[3] * u.solMass     # Hot CGM mass
        m_cgm_cold = mass[4] * u.solMass    # Cold CGM mass
        m_cgm = m_cgm_hot + m_cgm_cold      # Total CG
        m_metals = mass[5] * u.solMass
        m_halo = mass[6] * u.solMass        # total halo mass
        e_ism_wind = mass[7] * u.erg        # Energy gained from energy-loaded galactic winds
        e_cgm_cool = mass[8] * u.erg        # Energy loss from gas precipitation onto the Galaxy
        e_cgm_out = mass[9] * u.erg         # Energy loss from mass ejected from the CGM into IGM
        e_cgm_in = mass[10] * u.erg         # Energy gained from mass accretion from the IGM
        e_cgm = mass[11] * u.erg            # Total CGM energy
        m_bh = mass[12] * u.solMass        # Total black hole mass
        e_bh_thermfeedback = mass[13] * u.erg # energy output by BH
        # fmt: on

        # halo (derived) properties
        halo_rvir = virial_radius(z, m_halo).to(u.kpc)
        r1 = self.r_cgm_scale * halo_rvir  # our definition of inner radius of CGM
        halo_vcirc = circular_velocity(m_halo, halo_rvir).to(u.km / u.s)
        halo_vir_temp = virial_T(m_halo, halo_rvir).to(u.K)

        # overwrite mass loading as you are stepping through the ODE, no longer constant
        # self.eta_e = custom_energy_loading(m_halo, alpha=-0.5)
        # self.eta_m = custom_mass_loading(m_halo, alpha=-0.7)

        self.eta_m = vcirc_mass_loading(halo_vcirc, alpha_m=0.1)
        self.eta_e = vcirc_energy_loading(halo_vcirc, alpha_e=0.1)
        if self.eta_e > 1:  # physical
            self.eta_e = 1

        # e_cgm_hot ~ e_cgm
        cgm_temp = ((e_cgm / m_cgm_hot) * (self.mu / self.kb)).to(u.K)

        # dictates ISM depletion, old model, before KS law, deprected now
        t_depletion = depletion_time(z, m_star, self.exp, self.dep_time_norm)
        t_dynamical = t_ff(halo_rvir, halo_rvir, m_halo).to(u.Gyr)

        # rho_crit = LCDM.critical_density(z)
        cgm_metallicity = m_metals / m_cgm  # spread out over the CGM
        cgm_metallicity_sol = cgm_metallicity / self.Z_sol

        ######           cooling function value at this timestep
        ## Weirsama cooling
        # cooling_lambda = self.cooling_fn(
        #     (-4, np.log10(halo_vir_temp.value), cgm_metallicity_sol, 0)
        # ) * (u.erg * u.cm**3 * u.s**-1)
        ## basic cooling, to z dependence
        cooling_lambda = self.cooling_fn.cooling_function(
            cgm_temp.value, cgm_metallicity_sol
        ) * (u.erg * u.cm**3 * u.s**-1)

        if cooling_lambda < 0:
            print("cooling_lambda", cooling_lambda)
            cooling_lambda = 0 * (u.erg * u.cm**3 * u.s**-1)
            # raise ValueError("Negative cooling_lambda")

        # compute density normalization for power-law density model from CGM mass
        # using the m_cgm_hot instead of intire mass, lower density
        rho0 = density0(mCGM=m_cgm_hot, r0=r1, Rvir=halo_rvir, alpha=self.alpha)

        # estimate energy ejection loss timescale and limitit to dynamical time
        # c_sound = np.sqrt(e_cgm / m_cgm)  # approximate sound speed
        c_sound = np.sqrt(e_cgm / m_cgm_hot)
        t_ejection = (halo_rvir / c_sound).to(u.Gyr)  # ejection time of hot gas
        t_ejection = min(
            max(t_ejection.value, self.t_eject_lim_norm * t_dynamical.value),
            t_dynamical.value,
        )  # limited, guess needs to be callibrated
        t_ejection = t_ejection * u.Gyr

        # m_cgm_hot is used-- energy excess, how much there would be if the gas was at Tvir
        dot_e_cgm_out = (
            max(e_cgm - self.kb * halo_vir_temp * m_cgm_hot / self.mu, 0.0 * u.erg)
            / t_ejection
        ).to(u.erg / u.Gyr)

        #####  radiative losses in the cgm, integrated, Eq. 3 from Carr 2023
        dot_e_cgm_hot_loss = energy_loss(
            Lamb=cooling_lambda,
            Rvir=halo_rvir,
            r1=r1,
            rho0=rho0,  # new rho0
            mu=self.mu,
            alpha=self.alpha,
        ).to(u.erg / u.Gyr)

        #####  new galaxy profile
        # mcgm_and_ism = m_cgm + m_ism
        # Mbaryon = m_cgm + m_ism + m_star
        # total_Mgas = m_cgm + m_ism
        # dot_e_cgm_cool_new, _ = scipy.integrate.quad(
        #     makino_cooling_intergrand,
        #     r1.to(u.cm).value,  # r_vir to cm for integration limit
        #     halo_rvir.to(u.cm).value,  # r_vir to cm for integration limit
        #     args=(total_Mgas, Mbaryon, m_halo, z, cgm_temp.value, cooling_lambda, 200),
        # ) * (u.erg / u.s)
        # ### end new profile
        # ### print("cooling", dot_e_cgm_cool, dot_e_cgm_cool_new.to(u.erg / u.Gyr))
        # dot_e_cgm_hot_loss = dot_e_cgm_cool_new.to(u.erg / u.Gyr)

        # using the effective temp of CGM, get the effective energy, using only the hot gas
        cgm_specific_e = self.cgm_ejecti_specific_energy_ratio * max(
            e_cgm / m_cgm_hot, self.kb * cgm_temp / self.mu
        )
        # (effective) cooling time of CGM, specific energy is the energy per unit mass
        tcool = (cgm_specific_e / dot_e_cgm_hot_loss) * m_cgm_hot
        tcool_real = tcool.to(u.Gyr)

        # XXX: include Compton cooling?

        # XXX: Deprecated, but included for legacy plotting
        tcool_grav = tcool_real + self.grav_cooling_time_norm * t_dynamical

        # the real cooling time of the CGM, arbitrarily set by cooling plus some dynamical time
        tcool_eff = tcool_real + self.cooling_time_norm * t_dynamical  # XXX: Deprecated

        # the cooling time of the CGM, the energy of the CGM resides in the hot gas
        dot_e_cgm_cooling = (e_cgm / tcool_real).to(u.erg / u.Gyr)

        ##################### moving on to mass evolution

        # CGM hot gas mass loss due to cooling, 2 phase
        dot_m_cgm_hot_cooling = (m_cgm_hot / tcool_real).to(u.solMass / u.Gyr)
        dot_m_cgm_cold_falling = (m_cgm_cold / t_dynamical).to(u.solMass / u.Gyr)
        dot_m_sfr = m_ism / t_depletion
        if t_depletion < 0:
            raise ValueError("Negative depletion time")
        # if e_cgm < 0:
        #     raise ValueError("Negative CGM energy")
        # # star formation rate using sigma, KS law, adopted 	arXiv:2309.07957 Vallini
        r_disk = self.disk_scale * halo_rvir.value  # kpc

        sigma0 = m_ism.value / (2 * np.pi * r_disk**2)  # msun / kpc^2
        Asfr = 1e-12 * self.ks_kappa_s * 1e9  # msun / Gyr / kpc^2
        dot_m_star = (
            Asfr * sigma0**self.ks_n * (2 * np.pi * r_disk**2) / self.ks_n**2
        )  # msun / Gyr
        ############
        dot_m_sfr = dot_m_star * (u.solMass / u.Gyr)
        ############

        # dot_m_bulge = (4 * np.pi * Asfr * sigma0**n) / n**3
        # dot_m_bulge *= (2 * r_disk**2 - np.exp(-n * r_bulge / r_disk) * (
        #     2 * r_disk**2 + 2 * n * r_bulge * r_disk**2 + n**2 * r_bulge**2 * r_disk
        # ))

        dot_m_bulge = (2 * np.pi * Asfr * sigma0**self.ks_n) / self.ks_n**2
        dot_m_bulge *= r_disk**2 - np.exp(-self.ks_n * self.r_bulge / r_disk) * (
            r_disk**2 + self.ks_n * self.r_bulge * r_disk
        )
        # print(halo_rvir, r_disk)
        dot_mstar_bulge = dot_m_bulge * (u.solMass / u.Gyr)  # msun / Gyr
        # print( dot_mstar_bulge,  dot_m_sfr )
        # total ISM mass rate of change
        dot_m_ism = (dot_m_cgm_cold_falling - dot_m_sfr * (1 + self.eta_m)).to(
            u.solMass / u.Gyr
        )
        # star formation rate, as above
        dot_m_sfr = dot_m_sfr.to(u.solMass / u.Gyr)
        # halo inflall rate
        dot_m_halo = halo_infall_fakhouri(z, m_halo)
        # consider only baryonic mass
        dot_m_cgm_in = self.fb * dot_m_halo  # eq. 6

        # NOTE: the gas that gets ejected should be the most energetic gas, CGM eject loss term, eq 10
        dot_m_cgm_out = ((1 / cgm_specific_e) * dot_e_cgm_out).to(u.solMass / u.Gyr)

        ####### calculate the energies that depend on the mass changes

        # ratio of dot ECGM_in / ECGM_out
        dot_energy_from_infall = (self.kb * halo_vir_temp / self.mu).to(
            u.erg / u.solMass
        ) * dot_m_cgm_in
        e_ejection_to_infall_ratio = self.cgm_infall_prevention_const * (
            dot_energy_from_infall / dot_e_cgm_out
        )
        f_prevent = min(max(e_ejection_to_infall_ratio, 0.1), 1.0)  # 0.1 < f < 1
        # f_prevent = min(max(e_ejection_to_infall_ratio, 0.2), 1.0)  # 0.2 < f < 1
        # f_prevent =1
        dot_m_cgm_in *= f_prevent

        if self.verbose:
            print(
                f"dot_m_cgm_in: {dot_m_cgm_in:.2e}, f_prevent: {f_prevent:.2e}, -dot_e_cgm_out: {dot_e_cgm_out:.2e}, -dot_energy_from_infall: {dot_energy_from_infall:.2e}, -e_ejection_to_infall_ratio: {e_ejection_to_infall_ratio:.2e}, --max_ratio: {max(e_ejection_to_infall_ratio, 0.1):.2e}"
            )

        ##################### BH routines
        c_s = np.sqrt(self.kb * cgm_temp / self.mu).to(u.km / u.s)
        rho_0_hot = rho0
        dot_m_bh_bondi = (4 * np.pi * self.G**2 * rho_0_hot * m_bh**2) / c_s**3
        dot_m_bh_bondi = self.bondi_boost * dot_m_bh_bondi.to(u.solMass / u.Gyr)
        
        r_bhgrav = 2  # ckpc / h
        scale_fac = 1 / (1 + z)
        # r_grav_physical = r_bhgrav * (scale_fac / h)
        r_grav_physical = 1 # kpc

        # find the gas the BH accretion kernel by integrating the profile to r_grav_physical, analytical solution
        m_gas_in_bh_disk = (
            2
            * np.pi
            * sigma0
            * (
                r_disk**2
                - (r_disk * r_grav_physical + r_disk**2)
                * np.exp(-r_grav_physical / r_disk)
            )
        )
       
        #### bulge definition
        effective_bulge_mass = np.max([m_bulge.value - (np.pi * 1e9), 0])
        m_star_in_bh_disk  = m_bulge.value - effective_bulge_mass
        m_enclosed =  m_gas_in_bh_disk + m_star_in_bh_disk + effective_bulge_mass
        mbh_disk =m_gas_in_bh_disk + m_star_in_bh_disk # the disk component
        f_gas = m_gas_in_bh_disk /  mbh_disk
        f_disk = mbh_disk / m_enclosed
        f_0 = 0.31 * f_disk**2 * (mbh_disk / 1e9) ** -0.666

        if mbh_disk > 0:
            dot_m_tla = (
                0.15
                * f_disk**2.5
                * (m_bh.value / 1e8) ** (1 / 6)
                * (m_enclosed/ 1e9)
                * (r_grav_physical / 0.1) ** (-3 / 2)
                * (1 + f_0 / f_gas) ** -1
            )
        else:
            dot_m_tla = 0.0
        dot_m_bh_tla = dot_m_tla * (u.solMass / u.yr)
        dot_m_bh_tla = dot_m_bh_tla.to(u.solMass / u.Gyr)
        dot_m_bh =  dot_m_bh_bondi + dot_m_bh_tla

        dot_e_bh_thermfeedback = (self.ep_bh_feedback_eff * self.ep_bh_radeff * dot_m_bh * consts.c**2).to(
            u.erg / u.Gyr
        )
        dot_m_bh *= (1 - self.ep_bh_radeff)

        ##################### BH routines end

        # energy input from SF
        dot_e_ism_wind = energy_gain(self.eta_e, dot_m_sfr)
        # energy due to accretion, eq 16
        dot_e_cgm_in = self.dbug_norm_for_accretion_energy_rate *  f_prevent * (
            self.kb * halo_vir_temp / self.mu * dot_m_cgm_in
        ).to(u.erg * u.Gyr**-1)
        # CGM feedback gain term, eq 9
        dot_m_ism_wind = dot_m_sfr * self.eta_m

        ####  main derivative
        dot_m_cgm_hot = (
            dot_m_cgm_in + dot_m_ism_wind - dot_m_cgm_hot_cooling - dot_m_cgm_out
        )
        # print("dot_m_cgm_hot", dot_m_cgm_hot)
        dot_m_cgm_cold = dot_m_cgm_hot_cooling - dot_m_cgm_cold_falling
        # print("dot_m_cgm_cold", dot_m_cgm_cold, "dot_m_cgm_hot_cooling", dot_m_cgm_hot_cooling, "dot_m_cgm_cold_falling", dot_m_cgm_cold_falling)

        dot_e_cgm = (
            dot_e_ism_wind
            + 1 * dot_e_cgm_in
            - dot_e_cgm_out
            - 1 * dot_e_cgm_hot_loss
            + dot_e_bh_thermfeedback
        )
        dot_m_metal = (
            self.metal_yield * self.eta_z * dot_m_sfr
            + self.Z_IGM * self.Z_sol * dot_m_cgm_in
            - cgm_metallicity * (dot_m_cgm_hot_cooling + dot_m_cgm_out)
        )

        ##TODO: set mdot to 0 smoothly, instead of hard limit
        if (m_cgm_hot.value < 5e3) & (dot_m_cgm_hot.value < 0):
            # print("dot_m_cgm_hot < 0, correcting", dot_m_cgm_hot)
            dot_m_cgm_hot *= max((m_cgm_hot.value - 5e3) / 5e3, 0)

        if self.verbose:
            print(" ")
            print("**time", t, "z", z)
            if dot_m_cgm_cold_falling < dot_m_sfr:
                print(
                    "dot_m_cgm_cold_falling < dot_m_sfr",
                    dot_m_cgm_cold_falling,
                    dot_m_sfr,
                )
            print("> m_ism, m_star, m_cgm_hot, m_metals, m_halo")
            print(
                ">>  {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    m_ism, m_star, m_cgm_hot, m_metals, m_halo
                )
            )
            print("> e_ism_wind, e_cgm_cool, e_cgm_out, e_cgm_in, e_cgm")
            print(
                ">>  {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    e_ism_wind, e_cgm_cool, e_cgm_out, e_cgm_in, e_cgm
                )
            )
            print("pre limited ot_m_cgm_hot{:.2e}".format(dot_m_cgm_hot))

        halo_sfe = m_star / (m_halo * self.fb)

        if self.verbose:
            # do some checks
            print("mass of the CGM: {:.2e}".format(m_cgm_cold + m_cgm_hot))
            print("post-limited dot_m_cgm_hot: {:.2e}".format(dot_m_cgm_hot))
            print("dot m CGM in: {:.2e}".format(dot_m_cgm_in))
            print("dot m ism wind: {:.2e}".format(dot_m_ism_wind))
            print("dot m cgm cool: {:.2e}".format(dot_e_cgm_cooling))
            print("dot m cgm out: {:.2e}".format(dot_m_cgm_out))
            # print all timescales
            print("t_depletion: {:.2e}".format(t_depletion))
            print("t_dynamical: {:.2e}".format(t_dynamical))
            print("tcool_eff: {:.2e}".format(tcool_grav.to(u.Gyr)))
            print("t_ejection: {:.2e}".format(t_ejection))
            print("t_ff: {:.2e}".format(t_ff(halo_rvir, halo_rvir, m_halo).to(u.Gyr)))
            # print all the dot_ value
            print("> dot_m_ism, dot_m_sfr, dot_m_cgm, dot_m_metal, dot_m_halo")
            print(
                ">> {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    dot_m_ism, dot_m_sfr, dot_m_cgm_cold, dot_m_metal, dot_m_halo
                )
            )
            print(
                "> dot_e_ism_wind, dot_e_cgm_cool, dot_e_cgm_out, dot_e_cgm_in, dot_e_cgm"
            )
            print(
                ">>  {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    dot_e_ism_wind,
                    dot_e_cgm_cooling,
                    dot_e_cgm_out,
                    dot_e_cgm_in,
                    dot_e_cgm,
                )
            )

        # if dot_m_sfr is negative, stop
        # if m_ism < 0:
        #     raise ValueError("Negative ISM mass")
        if ode_mode:
            derivs = np.array(
                [
                    dot_m_ism.value,
                    dot_m_sfr.value,
                    dot_mstar_bulge.value,
                    dot_m_cgm_hot.value,
                    dot_m_cgm_cold.value,
                    dot_m_metal.value,
                    dot_m_halo.value,
                    dot_e_ism_wind.value,
                    dot_e_cgm_cooling.value,
                    dot_e_cgm_out.value,
                    dot_e_cgm_in.value,
                    dot_e_cgm.value,
                    dot_m_bh.value,
                    dot_e_bh_thermfeedback.value,
                ]
            )
            return derivs
        else:
            return np.array(
                [
                    halo_vir_temp.value,
                    dot_e_cgm_out.value,
                    dot_e_cgm_cooling.value,
                    dot_e_cgm_in.value,
                    dot_e_ism_wind.value,
                    dot_m_cgm_out.value,
                    dot_m_cgm_hot.value,  # hot bucket that feeds the cold buckets
                    dot_m_cgm_cold.value,  # cold bucket that feeds the ISM
                    dot_m_cgm_in.value,
                    dot_m_ism_wind.value,
                    f_prevent,
                    halo_sfe,
                    t_depletion.value,
                    t_dynamical.value,
                    tcool_real.value,
                    tcool_grav.value,
                    tcool_eff.value,
                    t_ejection.value,
                    cooling_lambda.value,
                    halo_rvir.value,
                    dot_m_sfr.value,
                    dot_mstar_bulge.value,
                    dot_m_bh.value,
                    dot_e_bh_thermfeedback.value,
                    dot_m_bh_bondi.value,
                    dot_m_bh_tla.value,
                    m_gas_in_bh_disk,
                    m_star_in_bh_disk,
                    effective_bulge_mass,
                ]
            )

    def run_halo(self):
        print("self.time_interval", self.time_interval, "tstep", self.tstep)
        mhalo_t0 = initial_mhalo(self.mhalo_z0, self.time_interval)

        # mhalo_t0 = self.mhalo_init.value
        rvir = virial_radius(z=self.zfinal, mhalo=mhalo_t0 * u.solMass).to(u.kpc)
        tvir = virial_T(mhalo_t0 * u.solMass, rvir).to(u.K)

        print(
            "Initial halo mass (t = {:.2f} Gyr) = {:.2e} Msol".format(
                self.time_interval[0], mhalo_t0
            )
        )
        mass_ism_gas_0 = 1e3
        mass_star_0 = 1e3
        mass_bulge_0 = mass_star_0  #
        mass_cgm_hot_0 = 1e3
        mass_cgm_cold_0 = 1e3
        mass_cgm_0 = mass_cgm_hot_0 + mass_cgm_cold_0
        initial_cgm_metal_zsun = 0.001
        # get the corresponding metal mass
        mass_cgm_metals_0 = mass_cgm_0 * initial_cgm_metal_zsun * self.Z_sol
        e_cgm_0 = (mass_cgm_hot_0 * u.solMass * self.fb * self.kb * tvir / self.mu).to(
            u.erg
        )
        e_ism_wind_0 = 1e3 * u.erg
        e_cgm_cooling_0 = 1e3 * u.erg
        e_cgm_out_0 = 1e3 * u.erg
        e_cgm_in_0 = 1e3 * u.erg

        m_bh_seed = 1e2 * u.solMass
        e_bh_feedback_0 = 0.0 * u.erg
        # initial conditions masses and energ
        masses_initial = np.array(
            [
                mass_ism_gas_0,
                mass_star_0,
                mass_bulge_0,
                mass_cgm_hot_0,
                mass_cgm_cold_0,
                mass_cgm_metals_0,
                mhalo_t0,
                e_ism_wind_0.value,
                e_cgm_cooling_0.value,
                e_cgm_out_0.value,
                e_cgm_in_0.value,
                e_cgm_0.value,
                m_bh_seed.value,
                e_bh_feedback_0.value,
            ]
        )
        print(
            "> initial values ISM  = {:.2e} Msol \t Stellar mass = {:.2e} Msol\tBulge stellar mass = {:.2e} Msol\tCGM hot mass = {:.2e} Msol\tCGM cold mass = {:.2e} Msol\tMetal mass = {:.2e} Msol\tCGM metallicity = {:.4f} (Zsun)\tHalo mass = {:.2e} Msol\tISM wind energy = {:.2e} erg\tCGM cooling energy = {:.2e} erg\tCGM out energy = {:.2e} erg\tCGM in energy = {:.2e} erg\tCGM energy = {:.2e} erg".format(
                masses_initial[0],
                masses_initial[1],
                masses_initial[2],
                masses_initial[3],
                masses_initial[4],
                masses_initial[5],
                (masses_initial[5] / mass_cgm_0) / self.Z_sol,
                masses_initial[6],
                masses_initial[7],
                masses_initial[8],
                masses_initial[9],
                masses_initial[10],
                masses_initial[11],
            )
        )

        # print(
        #     "> initial values ISM  = {:.2e} Msol \t Stellar mass = {:.2e} Msol\tBulge stellar mass = {:.2e} Msol\tCGM hot mass = {:.2e} Msol\tCGM cold mass = {:.2e} Msol\tMetal mass = {:.2e} Msol\tCGM metallicity = {:.4f} (Zsun)\tHalo mass = {:.2e} Msol\tISM wind energy = {:.2e} erg\tCGM cooling energy = {:.2e} erg\tCGM out energy = {:.2e} erg\tCGM in energy = {:.2e} erg\tCGM energy = {:.2e} erg".format(
        #     masses_initial[0],
        #     masses_initial[1],
        #     masses_initial[2],
        #     masses_initial[3],
        #     masses_initial[4],
        #     masses_initial[5],
        #     (masses_initial[5] / mass_cgm_0) / self.Z_sol,
        #     masses_initial[6],
        #     masses_initial[7],
        #     masses_initial[8],
        #     masses_initial[9],
        #     masses_initial[10],
        #     )
        # )

        if self.tstep != 1:
            t = self.evaluation_time_array
            solution = solve_ivp(
                self.mass_evolution,
                self.time_interval,
                masses_initial,
                # method="RK45",
                # rtol=1e-5,
                t_eval=t,
            )
        else:
            solution = solve_ivp(
                self.mass_evolution,
                self.time_interval,
                masses_initial,
            )

        adaptive_tsteps = solution.t
        adaptive_z = cosmology.z_at_value(LCDM.age, adaptive_tsteps * u.Gyr)

        #### solutions
        mgas_t = solution.y[0]
        mstar_t = solution.y[1]
        mass_bulge_t = solution.y[2]
        # mcgm_t = solution.y[2]
        mcgm_hot_t = solution.y[3]
        mcgm_cold_t = solution.y[4]
        mcgm_t = mcgm_hot_t + mcgm_cold_t  # total CGM mass
        mmetals_t = solution.y[5]
        mhalo_t = solution.y[6]

        egy_t = solution.y[7]  # energy gained from energy-loaded galactic winds
        egy_radloss_t = solution.y[8]  # energy lost due to cooling
        egy_eject_t = solution.y[9]  # energy ejected from the CGM
        egy_accrete_t = solution.y[10]  # energy accreted from the IGM
        egy_cgm_t = solution.y[11]  # total cgm energy

        metal_cgm_mass = mmetals_t / mcgm_t  # CGM metallicity ratio
        metal_cgm_mass_sol = metal_cgm_mass / self.Z_sol

        # BH
        mbh = solution.y[12]
        e_bh = solution.y[13]

        # CGM metallicity ratio in solar units
        # diagnose the solution status
        print(
            "*** Solution status: {} (0: succes, -1: step failed, 1: termination )".format(
                solution.status
            )
        ),
        # print("Number of function evaluations: ", solution.nfev)

        # the final values
        print(
            f"Final halo mass = {mhalo_t[-1]:.2e} Msol\t"
            f"Initial halo mass = {mhalo_t[0]:.2e} Msol\t"
            f"Final stellar mass = {mstar_t[-1]:.2e} Msol\t"
            f"Peak halo scale efficiency = {np.max(mstar_t / (mhalo_t * self.fb)):.2e}\t"
            f"Final CGM mass = {mcgm_t[-1]:.2e} Msol\t"
            f"Final CGM metallicity = {metal_cgm_mass_sol[-1]:.2e}\t"
            f"Final CGM energy = {egy_cgm_t[-1]:.2e} erg"
        )

        # fill the ode_results with the results
        self.ode_results["z"] = adaptive_z
        self.ode_results["t"] = adaptive_tsteps
        self.ode_results["m_ism"] = mgas_t
        self.ode_results["m_star"] = mstar_t
        self.ode_results["m_bulge"] = mass_bulge_t
        self.ode_results["m_cgm"] = mcgm_t
        self.ode_results["m_cgm_hot"] = mcgm_hot_t
        self.ode_results["m_cgm_cold"] = mcgm_cold_t
        self.ode_results["m_metals"] = mmetals_t
        self.ode_results["m_halo"] = mhalo_t
        self.ode_results["egy_ism_wind"] = egy_t
        self.ode_results["egy_radloss"] = egy_radloss_t
        self.ode_results["egy_eject"] = egy_eject_t
        self.ode_results["egy_accrete"] = egy_accrete_t
        self.ode_results["egy_cgm"] = egy_cgm_t
        self.ode_results["metal_cgm_mass"] = metal_cgm_mass
        self.ode_results["metal_cgm_mass_sol"] = metal_cgm_mass_sol
        self.ode_results["m_bh"] = mbh
        self.ode_results["egy_bh"] = e_bh

    def get_results(self):
        """
        main results of the ODE
        """
        return self.ode_results

    def get_derived_quantities(self):
        """get either some of the derived quantities or derivatives

        Returns:
            _type_: _description_
        """

        # the differnce between this and the one above is that this is dynamically updated, while the resulsta are from IVP

        ## we can calculate the derived quantities from the results
        t = self.ode_results["t"]

        mgas_and_energy_array = np.array(
            [
                self.ode_results["m_ism"],
                self.ode_results["m_star"],
                self.ode_results["m_bulge"],
                self.ode_results["m_cgm_hot"],
                self.ode_results["m_cgm_cold"],
                self.ode_results["m_metals"],
                self.ode_results["m_halo"],
                self.ode_results["egy_ism_wind"],
                self.ode_results["egy_radloss"],
                self.ode_results["egy_eject"],
                self.ode_results["egy_accrete"],
                self.ode_results["egy_cgm"],
                self.ode_results["m_bh"],
                self.ode_results["egy_bh"],
            ]
        ).T

        # print(mgas_and_energy_array.shape)
        derived_quantities = []
        for i, time in enumerate(t):
            derived_quantities.append(
                self.mass_evolution(time, mgas_and_energy_array[i, :], ode_mode=False)
            )

        derived_quantities = np.array(derived_quantities)
        return {
            "sim_time": t,
            "z": cosmology.z_at_value(LCDM.age, np.array(t) * u.Gyr),
            "tvir": derived_quantities[:, 0],
            "dot_e_cgm_out": derived_quantities[:, 1],
            "dot_e_cgm_cooling": derived_quantities[:, 2],
            "dot_e_cgm_in": derived_quantities[:, 3],
            "dot_e_ism_wind": derived_quantities[:, 4],
            "dot_m_cgm_out": derived_quantities[:, 5],
            "dot_m_cgm_hot": derived_quantities[:, 6],  # cold gas falling into ISM
            "dot_m_cgm_cold": derived_quantities[:, 7],  # hot gas cooling to cold gas
            "dot_m_cgm_in": derived_quantities[:, 8],
            "dot_m_ism_wind": derived_quantities[:, 9],
            "f_prevent": derived_quantities[:, 10],
            "f_star": derived_quantities[:, 11],
            "t_dep": derived_quantities[:, 12],
            "t_dyn": derived_quantities[:, 13],
            "tcool_real": derived_quantities[:, 14],
            "tcool_grav": derived_quantities[:, 15],
            "tcool_eff": derived_quantities[:, 16],
            "t_ejection": derived_quantities[:, 17],
            "cooling_lambda": derived_quantities[:, 18],
            "halo_rvir": derived_quantities[:, 19],
            "dot_m_sfr": derived_quantities[:, 20],
            "dot_m_bulge": derived_quantities[:, 21],
            "dot_m_bh": derived_quantities[:, 22],
            "dot_e_bh_thermfeedback": derived_quantities[:, 23],
            "dot_m_bh_bondi": derived_quantities[:, 24],
            "dot_m_bh_tla": derived_quantities[:,25],
            "m_gas_in_bh_disk": derived_quantities[:, 26],
            "m_star_in_bh_disk": derived_quantities[:, 27],
            "effective_bulge_mass": derived_quantities[:, 28],
            
        }


def plot_halo_profile(results, derived_quant):
    t = derived_quant["sim_time"]
    tvir = derived_quant["tvir"]
    halo_rvir = derived_quant["halo_rvir"]
    mgas_t = results["m_ism"]
    mstar_t = results["m_star"]
    mcgm_t = results["m_cgm"]
    egy_cgm_t = results["egy_cgm"]
    mhalo_t = results["m_halo"]

    fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=300, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0)
    cmap = plt.get_cmap("cmr.tropical")
    norm_z = plt.Normalize(vmin=derived_quant["z"][-1], vmax=derived_quant["z"][0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_z)
    sm.set_array([])
    mu = 0.6 * consts.m_p
    kb = consts.k_B

    # for the standard power law
    for i, time in enumerate(t):
        r = np.linspace(0.1, 2, 100) * halo_rvir[i]
        r1 = halo_rvir[i] * 0.1
        rho0 = density0(mCGM=mcgm_t[i], r0=r1, Rvir=halo_rvir[i], alpha=1.4)
        ax[0].plot(
            r / halo_rvir[i],
            rho0 * (r / r1) ** (-1.4),
            color=cmap(norm_z(derived_quant["z"][i])),
            lw=1,
        )

    ax[0].set(
        xlabel="r / Rvir",
        ylabel=r"$\rho_{\rm CGM} ~[{\rm M_\odot kpc^{-3}}]$",
        yscale="log",
        xscale="log",
        xlim=(0.1, 2),
    )

    # for the updated profiles
    for i, time in enumerate(t):
        r = np.linspace(0.1, 2, 100) * halo_rvir[i] * u.kpc
        mass_of_cgm = mcgm_t[i] * u.solMass
        egy_cgm = egy_cgm_t[i] * u.erg
        T_eff = ((egy_cgm / mass_of_cgm) * (mu / kb)).to(u.K).value
        mbaryon = mgas_t[i] + mstar_t[i] + mcgm_t[i]
        mgas = mgas_t[i] + mcgm_t[i]

        rho = rho_gas_makino(
            r,
            mgas * u.Msun,
            mbaryon * u.Msun,
            mhalo_t[i] * u.Msun,
            derived_quant["z"][i],
            halo_temp=T_eff,
            Delc=200,
        ).to(u.Msun * u.kpc**-3)

        ax[1].plot(
            r / halo_rvir[i], rho, color=cmap(norm_z(derived_quant["z"][i])), lw=1
        )

    ax[1].set(
        xlabel="r / Rvir",
        yscale="log",
        xscale="log",
        xlim=(0.1, 2),
    )

    # text in the first panel lower left on the halo properties
    text = r"$z_{{\rm end}} = {:.2f}$ " "\t".format(results["z"][-1].value)
    text += r"$M_{{\rm halo}} = {:.2e} ~{{\rm M_\odot}}$".format(mhalo_t[-1])
    text += r"$M_{{\rm CGM}} = {:.2e} ~{{\rm M_\odot}}$" "\t".format(mcgm_t[-1])
    text += r"$\rm M_{{\rm \star}} = {:.2e} ~{{\rm M_\odot}}$" "\t".format(mstar_t[-1])
    text += r"$R_{{\rm vir}} = {:.0f} ~{{\rm kpc}}$" "\t".format(halo_rvir[-1])
    text += r"$T_{{\rm vir}} = {:.2e} ~{{\rm K}}$" "\t".format(tvir[-1])

    ax[0].text(
        0.0,
        1.02,
        text,
        transform=ax[0].transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.78])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_z)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(r"redshift $z$")

    plt.savefig("./figures/halo_profile_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def sci_notation_tex(x, precision=1):
    """
    convert a float to a LaTeX-formatted scientific notation string.
    ___
    ex: 2e5 -> "2 \\times 10^{5}"
    """
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = round(x / 10**exponent, precision)
    if coeff == 1:
        return r"10^{{{}}}".format(exponent)  # no need to show 1
    return r"{} \times 10^{{{}}}".format(coeff, exponent)


def halo_timescales(results, derived_quant, title: str):
    t = derived_quant["sim_time"]
    z_red = derived_quant["z"]
    tvir = derived_quant["tvir"]
    virial_rad = derived_quant["halo_rvir"]
    disk_radius = 0.02 * virial_rad  # assume disk radius is 10% of the virial radius

    dot_e_cgm_out = derived_quant["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_quant["dot_e_cgm_cooling"]
    dot_e_cgm_in = derived_quant["dot_e_cgm_in"]
    dot_e_ism_wind = derived_quant["dot_e_ism_wind"]
    dot_m_cgm_out = derived_quant["dot_m_cgm_out"]
    dot_m_cgm_hot = derived_quant["dot_m_cgm_hot"]  # hot gas cooling to cold gas
    dot_m_cgm_cold = derived_quant["dot_m_cgm_cold"]  # cold gas falling into ISM
    dot_m_cgm_in = derived_quant["dot_m_cgm_in"]
    dot_m_ism_wind = derived_quant["dot_m_ism_wind"]
    dot_m_sfr = derived_quant["dot_m_sfr"]
    dot_m_bulge = derived_quant["dot_m_bulge"]
    cooling_lambda = derived_quant["cooling_lambda"]
    f_prevent = derived_quant["f_prevent"]
    f_star = derived_quant["f_star"]

    t_depletion = derived_quant["t_dep"]

    t_dynamical = derived_quant["t_dyn"]
    tcool_real = derived_quant["tcool_real"]  # purely cooling time based on CGM
    tcool_eff = derived_quant["tcool_eff"]  # added some fraction of dynamical time
    tcool_grav = derived_quant["tcool_grav"]  # added dynamical time
    t_ejection = derived_quant["t_ejection"]

    adaptive_z = results["z"]
    t_adaptive = results["t"]
    mgas_t = results["m_ism"]
    mstar_t = results["m_star"]
    mass_bulge_t = results["m_bulge"]
    disk_stellar_mass = mstar_t - mass_bulge_t
    bulge_disk_ratio = mass_bulge_t / mstar_t
    mcgm_t = results["m_cgm"]
    mcgm_hot_t = results["m_cgm_hot"]
    mcgm_cold_t = results["m_cgm_cold"]
    mmetals_t = results["m_metals"]
    mhalo_t = results["m_halo"]
    egy_t = results["egy_ism_wind"]
    egy_radloss_t = results["egy_radloss"]
    egy_eject_t = results["egy_eject"]
    egy_accrete_t = results["egy_accrete"]
    egy_cgm_t = results["egy_cgm"]
    metal_cgm_mass = results["metal_cgm_mass"]
    metal_cgm_mass_sol = results["metal_cgm_mass_sol"]
    sigma0 = mgas_t / (np.pi * disk_radius**2)  # surface density of the ISM gas
    t_dep_eff = mgas_t / dot_m_sfr

    # derive the CGM energy change rate
    dot_egy_cgm = dot_e_cgm_in + dot_e_ism_wind - dot_e_cgm_cooling - dot_e_cgm_out

    dot_m_cgm_hot_cooling = mcgm_hot_t / tcool_real
    dot_m_cgm_cold_falling = mcgm_cold_t / t_dynamical
    # let's standardize the colors

    cmapset_2 = plt.get_cmap("Dark2")

    c_mcgm_hot = cmapset_2(3)
    c_mcgm_cold = cmapset_2(2)
    c_mcgm = cmapset_2(5)
    c_mism = cmapset_2(6)
    c_mstar = cmapset_2(4)
    c_mhalo = "grey"
    fb = 0.16  # baryon fraction
    run_text = (
        r"{:}"
        "\n"
        "$z$ = {:.2f} - {:.2f} ($t$ = {:.2f} - {:.2f} Gyr)"
        "\n"
        r"$M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $ M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$".format(
            title,
            adaptive_z[0].value,
            adaptive_z[-1].value,
            t[0],
            t[-1],
            adaptive_z[0].value,
            sci_notation_tex(mhalo_t[0]),
            adaptive_z[-1].value,
            sci_notation_tex(mhalo_t[-1]),
            0,
            sci_notation_tex(mhalo_at_z0(mhalo_t[-1] * u.Msun, adaptive_z[-1])),
        )
    )

    ########################################################
    ######################## the timescales, star formation efficiency, and the CGM temperature
    ########################################################
    fig, ax = plt.subplots(
        6,
        1,
        figsize=(5, 11),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 0.5, 0.5, 0.5, 0.5, 0.5]},
    )
    plt.subplots_adjust(hspace=0.0)
    ax = ax.flatten()
    ax[0].plot(t, tcool_real, label=r"$t_{\rm cool}$", color=c_mcgm_hot, lw=2)
    ax[0].plot(t, t_dynamical, label=r"$t_{\rm ff}$", color=c_mcgm_cold, lw=2)
    # effective depletion time
    ax[0].plot(t, t_dep_eff, label=r"$t_{\rm dep, eff}$", color=c_mstar, lw=2)
    ax[0].plot(
        t, t_depletion, label=r"$0.4 \times t_H  $", color="tab:blue", ls="--", lw=2
    )
    ax[0].plot(t, t_ejection, label=r"$t_{\rm ej}$", color=c_mhalo, lw=2)
    ax[0].set(ylabel=r"$\rm Timescales ~ [Gyr]$", yscale="log", ylim=(1e-3, 13))
    ax[0].legend(ncols=4, frameon=False, loc="upper right", fontsize=10)

    ax[1].plot(t, tvir, label=r"$T_{\rm vir}$", color=c_mhalo, lw=2)
    mu = 0.6 * consts.m_p
    kb = consts.k_B
    Teff = ((egy_cgm_t * u.erg) / (mcgm_hot_t * u.solMass) * (mu / kb)).to(u.K)
    ax[1].plot(t_adaptive, Teff, label=r"$T_{\rm eff}$", color=c_mcgm, lw=2)
    ax[1].set(
        ylabel=r"$\rm Temperature ~ [K]$",
        yscale="log",
        ylim=(2e3, max(Teff.max().value, tvir.max()) * 2),
    )
    ax[1].legend(ncols=2, frameon=False, loc="lower right", fontsize=10)

    # we can plot the cgm metallicity
    ax[2].plot(t, metal_cgm_mass_sol, color=c_mcgm, lw=2)
    ax[2].set(
        ylabel=r"$\rm CGM ~ metallicity ~ [Z_\odot]$",
        yscale="log",
        ylim=(5e-4, metal_cgm_mass_sol.max() * 2),
    )

    # we can plot the cooling lambda
    ax[3].plot(t, np.log10(cooling_lambda), color=c_mcgm, lw=2)
    ax[3].set(
        ylabel=r"log $\Lambda$ [erg cm$^3$ s$^{-1}$]",
        ylim=(np.log10(cooling_lambda).max() - 3, np.log10(cooling_lambda).max() + 1),
    )

    # plot the star formation efficiency
    ax[4].plot(t, f_star, color=c_mstar, lw=2)
    ax[4].set(
        ylabel=r"$f_\star [M_\star / M_{\rm halo} f_b]$",
        yscale="log",
        ylim=(1e-3, 2),
    )

    # plot the prevention factor
    ax[5].plot(t, f_prevent, color=c_mhalo, lw=2)
    ax[5].set(
        ylabel=r"$f_{\rm prevent}$", ylim=(0, 1.1), xlabel=r"${\rm time\: [Gyr]}$"
    )

    # make a twin redshift axis for the top row, using z
    # get the current x axis labels of the first row and their
    x_axis_tick_labels = ax[0].get_xticks()
    x_axis_tick_labels = np.array(x_axis_tick_labels)[2:-1]
    # prepend the lowest time value to the beginning of the array
    x_axis_tick_labels = np.insert(x_axis_tick_labels, 0, t[0])
    for i in range(1):
        # remove the minor ticks in the x axis only
        ax2 = ax[i].twiny()
        # make sure the ranges are the same
        ax2.plot(t, dot_e_cgm_out, color="k", alpha=0)
        ax2.set_xlim(t[0] * 0.9, t[-1])
        t_ticks = x_axis_tick_labels  # [8,  5,4, 3, 2, 1,  0.001]
        # z_ticks = np.geomspace(np.max(adaptive_z).value, np.min(adaptive_z).value, 5)
        # t_ticks = LCDM.age(z_ticks).value  # Convert redshift to corresponding time
        z_ticks = cosmology.z_at_value(LCDM.age, t_ticks * u.Gyr).value
        # ax2.set_xticks(x_axis_tick_labels)
        ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
        ax2.set_xlim(t[0] * 0.9, t[-1])
        ax2.set_xlabel(r"$z$")

    for a in ax:
        a.set_xlim(t[0] * 0.99, t[-1])

    ax[0].text(
        -0.10,
        1.15,
        run_text,
        transform=ax[0].transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    plt.show()


def halo_normalized_rates(results, derived_quant, title: str):
    t = derived_quant["sim_time"]
    z_red = derived_quant["z"]
    tvir = derived_quant["tvir"]
    virial_rad = derived_quant["halo_rvir"]
    disk_radius = 0.02 * virial_rad  # assume disk radius is 10% of the virial radius

    dot_e_cgm_out = derived_quant["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_quant["dot_e_cgm_cooling"]
    dot_e_cgm_in = derived_quant["dot_e_cgm_in"]
    dot_e_ism_wind = derived_quant["dot_e_ism_wind"]
    dot_m_cgm_out = derived_quant["dot_m_cgm_out"]
    dot_m_cgm_hot = derived_quant["dot_m_cgm_hot"]  # hot gas cooling to cold gas
    dot_m_cgm_cold = derived_quant["dot_m_cgm_cold"]  # cold gas falling into ISM
    dot_m_cgm_in = derived_quant["dot_m_cgm_in"]
    dot_m_ism_wind = derived_quant["dot_m_ism_wind"]
    dot_m_sfr = derived_quant["dot_m_sfr"]
    dot_m_bulge = derived_quant["dot_m_bulge"]
    cooling_lambda = derived_quant["cooling_lambda"]
    f_prevent = derived_quant["f_prevent"]
    f_star = derived_quant["f_star"]

    t_depletion = derived_quant["t_dep"]

    t_dynamical = derived_quant["t_dyn"]
    tcool_real = derived_quant["tcool_real"]  # purely cooling time based on CGM
    tcool_eff = derived_quant["tcool_eff"]  # added some fraction of dynamical time
    tcool_grav = derived_quant["tcool_grav"]  # added dynamical time
    t_ejection = derived_quant["t_ejection"]

    adaptive_z = results["z"]
    t_adaptive = results["t"]
    mgas_t = results["m_ism"]
    mstar_t = results["m_star"]
    mass_bulge_t = results["m_bulge"]
    disk_stellar_mass = mstar_t - mass_bulge_t
    bulge_disk_ratio = mass_bulge_t / mstar_t
    mcgm_t = results["m_cgm"]
    mcgm_hot_t = results["m_cgm_hot"]
    mcgm_cold_t = results["m_cgm_cold"]
    mmetals_t = results["m_metals"]
    mhalo_t = results["m_halo"]
    egy_t = results["egy_ism_wind"]
    egy_radloss_t = results["egy_radloss"]
    egy_eject_t = results["egy_eject"]
    egy_accrete_t = results["egy_accrete"]
    egy_cgm_t = results["egy_cgm"]
    metal_cgm_mass = results["metal_cgm_mass"]
    metal_cgm_mass_sol = results["metal_cgm_mass_sol"]
    sigma0 = mgas_t / (np.pi * disk_radius**2)  # surface density of the ISM gas
    t_dep_eff = mgas_t / dot_m_sfr

    # derive the CGM energy change rate
    dot_egy_cgm = dot_e_cgm_in + dot_e_ism_wind - dot_e_cgm_cooling - dot_e_cgm_out

    dot_m_cgm_hot_cooling = mcgm_hot_t / tcool_real
    dot_m_cgm_cold_falling = mcgm_cold_t / t_dynamical
    # let's standardize the colors

    cmapset_2 = plt.get_cmap("Dark2")

    c_mcgm_hot = cmapset_2(3)
    c_mcgm_cold = cmapset_2(2)
    c_mcgm = cmapset_2(5)
    c_mism = cmapset_2(6)
    c_mstar = cmapset_2(4)
    c_mhalo = "grey"
    fb = 0.16  # baryon fraction
    run_text = (
        r"{:}"
        "\n"
        "$z$ = {:.2f} - {:.2f} ($t$ = {:.2f} - {:.2f} Gyr)"
        "\n"
        r"$M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $ M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$".format(
            title,
            adaptive_z[0].value,
            adaptive_z[-1].value,
            t[0],
            t[-1],
            adaptive_z[0].value,
            sci_notation_tex(mhalo_t[0]),
            adaptive_z[-1].value,
            sci_notation_tex(mhalo_t[-1]),
            0,
            sci_notation_tex(mhalo_at_z0(mhalo_t[-1] * u.Msun, adaptive_z[-1])),
        )
    )
    ##############################################################################
    ########################## plotting normalized rates
    ##############################################################################
    fig, ax = plt.subplots(2, 1, figsize=(5, 6), dpi=200, sharex=True)
    ax[0].plot(
        t,
        (dot_m_sfr / dot_m_sfr.max()) + 0.1,
        label=r"$\dot{M}_\star$ + const",
        color=c_mstar,
        lw=1,
    )
    ax[0].plot(
        t,
        dot_m_cgm_hot_cooling / dot_m_cgm_hot_cooling.max(),
        label=r"$\dot{M}_{\rm CGM, hot \:cooling}$",
        color=c_mcgm_hot,
        lw=1,
    )
    ax[0].plot(
        t,
        dot_m_cgm_cold_falling / dot_m_cgm_cold_falling.max(),
        label=r"$\dot{M}_{\rm CGM, cold\: falling}$",
        color=c_mcgm_cold,
        lw=1,
        ls="-",
    )
    ax[0].plot(
        t,
        (dot_m_cgm_hot_cooling - dot_m_cgm_cold_falling) / dot_m_cgm_cold.max(),
        color="k",
        lw=1,
        label=r"$\dot{M}_{\rm CGM, cold}$",
    )
    ax[0].plot(
        t,
        dot_m_cgm_out / dot_m_cgm_out.max(),
        label=r"$\dot{M}_{\rm CGM, out}$",
        color=c_mcgm,
        lw=1,
    )
    ax[0].plot(
        t,
        dot_e_cgm_in / dot_e_cgm_in.max(),
        label=r"$\dot{E}_{\rm CGM, in}$",
        lw=1,
        color=c_mcgm_hot,
        ls="--",
    )
    ax[0].plot(
        t,
        dot_m_cgm_hot / dot_m_cgm_hot.max(),
        label=r"$\dot{M}_{\rm CGM, hot}$",
        color="red",
        lw=1,
        ls=":",
    )
    ax[0].set(ylabel=r"$\rm Mass ~ Rates ~ {\rm normalized}$")
    ax[0].legend(
        ncol=3, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.05)
    )
    # see if the Mcgm hot or cold are in phase or not
    hot_cgm_norm_rate_of_change = dot_m_cgm_hot / dot_m_cgm_hot.max()
    cold_cgm_norm_rate_of_change = dot_m_cgm_cold / dot_m_cgm_cold.max()
    phase_diff = hot_cgm_norm_rate_of_change - cold_cgm_norm_rate_of_change

    # not do the same but for dot_m_cgm_cold_falling and dot_m_cgm_hot_cooling
    hot_cgm_norm_cooling_rate_of_change = (
        dot_m_cgm_hot_cooling / dot_m_cgm_hot_cooling.max()
    )
    cold_cgm_norm_falling_rate_of_change = (
        dot_m_cgm_cold_falling / dot_m_cgm_cold_falling.max()
    )
    phase_diff_cooling_falling = (
        hot_cgm_norm_cooling_rate_of_change - cold_cgm_norm_falling_rate_of_change
    )

    ax[1].plot(
        t,
        phase_diff,
        label=r"$\dot{M}_{\rm CGM, hot} - \dot{M}_{\rm CGM, cold}$",
        color="k",
        lw=1,
    )
    ax[1].plot(
        t,
        phase_diff_cooling_falling,
        label=r"$\dot{M}_{\rm CGM, hot\: cooling} - \dot{M}_{\rm CGM, cold\: falling}$",
        color="tab:blue",
        lw=1,
    )
    ax[1].set(ylabel="difference of normalized values", xlabel=r"${\rm time\: [Gyr]}$")
    ax[1].axhline(0, color="k", lw=0.5, ls="--")
    ax[1].legend(
        ncol=2, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.05)
    )

    plt.show()


def halo_detailed_energy_and_mass_plots(results, derived_quant, title: str):
    ##############################################################################
    ########################## compute the ratios that go into the dot E CGM
    ##############################################################################
    t = derived_quant["sim_time"]
    z_red = derived_quant["z"]
    tvir = derived_quant["tvir"]
    virial_rad = derived_quant["halo_rvir"]
    disk_radius = 0.02 * virial_rad  # assume disk radius is 10% of the virial radius

    dot_e_cgm_out = derived_quant["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_quant["dot_e_cgm_cooling"]
    dot_e_cgm_in = derived_quant["dot_e_cgm_in"]
    dot_e_ism_wind = derived_quant["dot_e_ism_wind"]
    dot_m_cgm_out = derived_quant["dot_m_cgm_out"]
    dot_m_cgm_hot = derived_quant["dot_m_cgm_hot"]  # hot gas cooling to cold gas
    dot_m_cgm_cold = derived_quant["dot_m_cgm_cold"]  # cold gas falling into ISM
    dot_m_cgm_in = derived_quant["dot_m_cgm_in"]
    dot_m_ism_wind = derived_quant["dot_m_ism_wind"]
    dot_m_sfr = derived_quant["dot_m_sfr"]
    dot_m_bulge = derived_quant["dot_m_bulge"]
    cooling_lambda = derived_quant["cooling_lambda"]
    f_prevent = derived_quant["f_prevent"]
    f_star = derived_quant["f_star"]

    t_depletion = derived_quant["t_dep"]

    t_dynamical = derived_quant["t_dyn"]
    tcool_real = derived_quant["tcool_real"]  # purely cooling time based on CGM
    tcool_eff = derived_quant["tcool_eff"]  # added some fraction of dynamical time
    tcool_grav = derived_quant["tcool_grav"]  # added dynamical time
    t_ejection = derived_quant["t_ejection"]

    adaptive_z = results["z"]
    t_adaptive = results["t"]
    mgas_t = results["m_ism"]
    mstar_t = results["m_star"]
    mass_bulge_t = results["m_bulge"]
    disk_stellar_mass = mstar_t - mass_bulge_t
    bulge_disk_ratio = mass_bulge_t / mstar_t
    mcgm_t = results["m_cgm"]
    mcgm_hot_t = results["m_cgm_hot"]
    mcgm_cold_t = results["m_cgm_cold"]
    mmetals_t = results["m_metals"]
    mhalo_t = results["m_halo"]
    egy_t = results["egy_ism_wind"]
    egy_radloss_t = results["egy_radloss"]
    egy_eject_t = results["egy_eject"]
    egy_accrete_t = results["egy_accrete"]
    egy_cgm_t = results["egy_cgm"]
    metal_cgm_mass = results["metal_cgm_mass"]
    metal_cgm_mass_sol = results["metal_cgm_mass_sol"]
    sigma0 = mgas_t / (np.pi * disk_radius**2)  # surface density of the ISM gas
    t_dep_eff = mgas_t / dot_m_sfr

    # derive the CGM energy change rate
    dot_egy_cgm = dot_e_cgm_in + dot_e_ism_wind - dot_e_cgm_cooling - dot_e_cgm_out

    dot_m_cgm_hot_cooling = mcgm_hot_t / tcool_real
    dot_m_cgm_cold_falling = mcgm_cold_t / t_dynamical
    # let's standardize the colors

    cmapset_2 = plt.get_cmap("Dark2")

    c_mcgm_hot = cmapset_2(3)
    c_mcgm_cold = cmapset_2(2)
    c_mcgm = cmapset_2(5)
    c_mism = cmapset_2(6)
    c_mstar = cmapset_2(4)
    c_mhalo = "grey"
    fb = 0.16  # baryon fraction
    run_text = (
        r"{:}"
        "\n"
        "$z$ = {:.2f} - {:.2f} ($t$ = {:.2f} - {:.2f} Gyr)"
        "\n"
        r"$M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $ M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$".format(
            title,
            adaptive_z[0].value,
            adaptive_z[-1].value,
            t[0],
            t[-1],
            adaptive_z[0].value,
            sci_notation_tex(mhalo_t[0]),
            adaptive_z[-1].value,
            sci_notation_tex(mhalo_t[-1]),
            0,
            sci_notation_tex(mhalo_at_z0(mhalo_t[-1] * u.Msun, adaptive_z[-1])),
        )
    )

    fig, ax = plt.subplots(3, 1, figsize=(4, 5), dpi=300, sharex=True)

    dot_e_cgm_in_ratio = dot_e_cgm_in / dot_egy_cgm
    dot_e_cgm_out_ratio = dot_e_cgm_out / dot_egy_cgm
    dot_e_ism_wind_ratio = dot_e_ism_wind / dot_egy_cgm
    dot_e_cgm_cooling_ratio = dot_e_cgm_cooling / dot_egy_cgm
    total_dot_egy_cgm = (
        dot_e_cgm_in + dot_e_ism_wind - dot_e_cgm_cooling - dot_e_cgm_out
    )

    ax[0].plot(
        t,
        dot_e_ism_wind_ratio,
        label=r"$\dot{E}_{\rm ISM, winds}$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
    )
    ax[0].plot(
        t,
        dot_e_cgm_cooling_ratio,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
    )
    ax[0].plot(
        t,
        dot_e_cgm_in_ratio,
        label=r"$\dot{E}_{\rm CGM, in}" r"$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    ax[0].plot(
        t,
        dot_e_cgm_out_ratio,
        label=r"$\dot{E}_{\rm CGM, out}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    ax[0].plot(t, total_dot_egy_cgm / dot_egy_cgm, color=c_mcgm, lw=1, alpha=0.7)
    ax[0].set(
        ylabel=r"fraction of $\dot{E}_{\rm CGM} ~ $", yscale="log", ylim=(1e-4, 1e3)
    )
    ax[0].legend(
        bbox_to_anchor=(1.25, 1.2),
        loc="lower left",
        ncol=4,
        fontsize=8,
    )

    ax[1].plot(
        t, dot_egy_cgm, label=r"$\dot{E}_{\rm CGM}$", color=c_mcgm, lw=1.25, alpha=0.7
    )
    ax[1].set(ylabel=r"$\dot{E}_{\rm CGM} ~ [{\rm erg \ Gyr^{-1}}]$")
    ax[1].axhline(0, color="k", lw=0.5, ls="--")

    ax[2].plot(t, egy_cgm_t, label=r"$E_{\rm CGM}$", color=c_mcgm, lw=1.25, alpha=0.7)
    ax[2].set(
        ylabel=r"$E_{\rm CGM} ~ [{\rm erg}]$",
        xlabel=r"${\rm time\: [Gyr]}$",
    )

    # make an inset for the first plot at the start
    xlim_inset = (t[0] * 0.99, t[0] * 2)
    inset_ax = ax[0].inset_axes([1.2, -1.25, 1.25, 2.5])  # [x, y, width, height]
    inset_ax.plot(
        t,
        dot_e_ism_wind_ratio,
        label=r"$\dot{E}_{\rm ISM, winds}$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
    )
    inset_ax.plot(
        t,
        dot_e_cgm_cooling_ratio,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
    )
    inset_ax.plot(
        t,
        dot_e_cgm_in_ratio,
        label=r"$\dot{E}_{\rm CGM, in}" r"$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    inset_ax.plot(
        t,
        dot_e_cgm_out_ratio,
        label=r"$\dot{E}_{\rm CGM, out}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    inset_ax.plot(t, total_dot_egy_cgm / dot_egy_cgm, color=c_mcgm, lw=1, alpha=0.7)
    inset_ax.set(
        ylabel=r"fraction of $\dot{E}_{\rm CGM} ~ $",
        xlim=xlim_inset,
        ylim=(1e-4, 1e3),
        yscale="log",
    )
    # mark inset
    ret = mark_inset(ax[0], inset_ax, loc1=2, loc2=2, zorder=0)
    # make a new inset axis for the timescales
    inset_ax2 = ax[2].inset_axes([1.2, 0, 1.25, 1])  # [x, y, width, height]
    inset_ax2.plot(
        t, t_dynamical, label=r"$t_{\rm ff}$", color=c_mcgm_cold, lw=1.25, alpha=1
    )
    inset_ax2.plot(
        t, tcool_real, label=r"$t_{\rm cool}$", color=c_mcgm_hot, lw=1.25, alpha=1
    )
    inset_ax2.plot(
        t, t_dep_eff, label=r"SF $t_{\rm dep, eff}$", color=c_mstar, lw=1.25, alpha=1
    )
    inset_ax2.plot(
        t, t_ejection, label=r"$t_{\rm ej}$", color=c_mhalo, lw=1.25, alpha=0.7
    )
    # inset_ax2.plot(t, egy_cgm_t / dot_egy_cgm, label=r"$E_{\rm CGM} / \dot{E}_{\rm CGM}$", color=c_mcgm, lw=1.25, alpha=0.7)
    inset_ax2.set(ylabel=r"$t ~ [Gyr]$", xlim=xlim_inset, yscale="log")
    # mark inset
    inset_ax2.legend(ncols=4, frameon=False)

    # add the run text to the top left corner
    ax[0].text(
        -0.15,
        1.15,
        run_text,
        transform=ax[0].transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    plt.show()
    ##############################################################################
    ########################### now do the same but for dot_m_cgm_hot
    ##############################################################################
    fig, ax = plt.subplots(3, 1, figsize=(4, 5), dpi=300, sharex=True)
    dot_m_cgm_in_ratio = dot_m_cgm_in / dot_m_cgm_hot
    dot_m_cgm_out_ratio = dot_m_cgm_out / dot_m_cgm_hot
    dot_m_ism_wind_ratio = dot_m_ism_wind / dot_m_cgm_hot
    dot_m_cgm_cooling_ratio = dot_m_cgm_hot_cooling / dot_m_cgm_hot
    total_dot_m_cgm_hot = (
        dot_m_cgm_in - dot_m_cgm_out + dot_m_ism_wind - dot_m_cgm_hot_cooling
    )

    ax[0].plot(
        t,
        dot_m_ism_wind_ratio,
        label=r"$\dot{M}_{\rm ISM, winds}$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
    )
    ax[0].plot(
        t,
        dot_m_cgm_cooling_ratio,
        label=r"$\dot{M}_{\rm CGM, cooling}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
    )
    ax[0].plot(
        t,
        dot_m_cgm_in_ratio,
        label=r"$\dot{M}_{\rm CGM, in}" r"$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    ax[0].plot(
        t,
        dot_m_cgm_out_ratio,
        label=r"$\dot{M}_{\rm CGM, out}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    ax[0].plot(t, total_dot_m_cgm_hot / dot_m_cgm_hot, color=c_mcgm, lw=1, alpha=0.7)
    ax[0].set(
        ylabel=r"fraction of $\dot{M}_{\rm CGM, hot} ~ $",
        yscale="log",
        ylim=(1e-4, 1e3),
    )
    ax[0].legend(
        bbox_to_anchor=(1.18, 1.1),
        loc="lower left",
        ncol=4,
        fontsize=8,
    )
    ax[1].plot(
        t,
        dot_m_cgm_hot,
        label=r"$\dot{M}_{\rm CGM, hot}$",
        color=c_mcgm_hot,
        lw=1.25,
        alpha=0.7,
    )
    ax[1].set(ylabel=r"$\dot{M}_{\rm CGM, hot} ~ [{\rm M_{\odot} \ Gyr^{-1}}]$")
    ax[1].axhline(0, color="k", lw=0.5, ls="--")
    ax[2].plot(
        t, mcgm_hot_t, label=r"$M_{\rm CGM, hot}$", color=c_mcgm_hot, lw=1.25, alpha=0.7
    )
    ax[2].set(
        ylabel=r"$M_{\rm CGM, hot} ~ [{\rm M_{\odot}}]$",
        xlabel=r"${\rm time\: [Gyr]}$",
    )

    # make an inset for the first plot at the start
    xlim_inset = (t[0] * 0.99, t[0] * 2)
    inset_ax = ax[0].inset_axes([1.2, -1.55, 1.25, 2.5])  # [x,     y, width, height]
    inset_ax.plot(
        t,
        dot_m_ism_wind_ratio,
        label=r"$\dot   {M}_{\rm ISM, winds}$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
    )
    inset_ax.plot(
        t,
        dot_m_cgm_cooling_ratio,
        label=r"$\dot {M}_{\rm CGM, cooling}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
    )
    inset_ax.plot(
        t,
        dot_m_cgm_in_ratio,
        label=r"$\dot{M}_{\rm CGM, in}" r"$",
        color="tab:red",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    inset_ax.plot(
        t,
        dot_m_cgm_out_ratio,
        label=r"$\dot{M}_{\rm CGM, out}$",
        color="tab:blue",
        lw=1.25,
        alpha=0.7,
        ls="--",
    )
    inset_ax.plot(t, total_dot_m_cgm_hot / dot_m_cgm_hot, color=c_mcgm, lw=1, alpha=0.7)
    inset_ax.set(
        ylabel=r"fraction of $\dot{M}_{\rm CGM, hot} ~ $",
        xlim=xlim_inset,
        ylim=(1e-4, 1e3),
        yscale="log",
    )
    # mark inset
    ret = mark_inset(ax[0], inset_ax, loc1=2, loc2=2)

    # now the text
    ax[0].text(
        -0.2,
        1.15,
        run_text,
        transform=ax[0].transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="left",
    )


def halo_diagnostics_v2(results, derived_quant, title: str):
    t = derived_quant["sim_time"]
    z_red = derived_quant["z"]
    tvir = derived_quant["tvir"]
    virial_rad = derived_quant["halo_rvir"]
    disk_radius = 0.02 * virial_rad  # assume disk radius is 10% of the virial radius

    dot_e_cgm_out = derived_quant["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_quant["dot_e_cgm_cooling"]
    dot_e_cgm_in = derived_quant["dot_e_cgm_in"]
    dot_e_ism_wind = derived_quant["dot_e_ism_wind"]
    dot_m_cgm_out = derived_quant["dot_m_cgm_out"]
    dot_m_cgm_hot = derived_quant["dot_m_cgm_hot"]  # hot gas cooling to cold gas
    dot_m_cgm_cold = derived_quant["dot_m_cgm_cold"]  # cold gas falling into ISM
    dot_m_cgm_in = derived_quant["dot_m_cgm_in"]
    dot_m_ism_wind = derived_quant["dot_m_ism_wind"]
    dot_m_sfr = derived_quant["dot_m_sfr"]
    dot_m_bulge = derived_quant["dot_m_bulge"]
    
    cooling_lambda = derived_quant["cooling_lambda"]
    f_prevent = derived_quant["f_prevent"]
    t_dynamical = derived_quant["t_dyn"]
    tcool_real = derived_quant["tcool_real"]  # purely cooling time based on CGM

    adaptive_z = results["z"]
    t_adaptive = results["t"]
    mgas_t = results["m_ism"]
    mstar_t = results["m_star"]
    mass_bulge_t = results["m_bulge"]
    mbulge_eff = derived_quant["effective_bulge_mass"]      
    disk_stellar_mass = mstar_t - mass_bulge_t
    bulge_disk_ratio = mass_bulge_t / mstar_t
    mcgm_t = results["m_cgm"]
    mcgm_hot_t = results["m_cgm_hot"]
    mcgm_cold_t = results["m_cgm_cold"]

    mhalo_t = results["m_halo"]
    egy_t = results["egy_ism_wind"]
    egy_radloss_t = results["egy_radloss"]
    egy_eject_t = results["egy_eject"]
    egy_accrete_t = results["egy_accrete"]
    egy_cgm_t = results["egy_cgm"]

    t_dep_eff = mgas_t / dot_m_sfr

    # derive the CGM energy change rate
    dot_egy_cgm = dot_e_cgm_in + dot_e_ism_wind - dot_e_cgm_cooling - dot_e_cgm_out

    dot_m_cgm_hot_cooling = mcgm_hot_t / tcool_real
    dot_m_cgm_cold_falling = mcgm_cold_t / t_dynamical

    # now do the BH
    mbh = results["m_bh"]
    egy_bh = results["egy_bh"]
    dot_m_bh = derived_quant["dot_m_bh"]
    dot_e_bh_thermfeedback = derived_quant["dot_e_bh_thermfeedback"]

    # custom bulge mass
    # r_bulge = 1
    # effective_bulge_mass = np.maximum(mass_bulge_t - (np.pi * 1e9), 0)
    # mask = effective_bulge_mass < np.pi * 1e9
    # effective_bulge_mass[mask] = 0
    # effective_bulge_mass[~mask] =   effective_bulge_mass[~mask] - (np.pi * 1e9)


    # effective_bulge_mass2 [effective_bulge_mass2< np.pi*1e9] =0
    # let's standardize the colors
    cmapset_2 = plt.get_cmap("Dark2")
    c_mcgm_hot = cmapset_2(3)
    c_mcgm_cold = cmapset_2(2)
    c_mcgm = cmapset_2(5)
    c_mism = cmapset_2(6)
    c_mstar = cmapset_2(4)
    c_mhalo = "grey"
    c_bh = "k"#cmapset_2(1)

    fb = 0.16  # baryon fraction
    run_text = (
        r"{:}"
        "\n"
        "$z$ = {:.2f} - {:.2f} ($t$ = {:.2f} - {:.2f} Gyr)"
        "\n"
        r"$M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $ M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$, $M_{{\rm h}}(z = {:.1f}) = {:} ~ M_\odot$".format(
            title,
            adaptive_z[0].value,
            adaptive_z[-1].value,
            t[0],
            t[-1],
            adaptive_z[0].value,
            sci_notation_tex(mhalo_t[0]),
            adaptive_z[-1].value,
            sci_notation_tex(mhalo_t[-1]),
            0,
            sci_notation_tex(mhalo_at_z0(mhalo_t[-1] * u.Msun, adaptive_z[-1])),
        )
    )

    ############################################################
    ################ now plot the halo halo physical properties
    ############################################################
    # fig, ax = plt.subplots(2, 1, figsize=(5, 4), dpi=300, sharex=True)
    # ax[0].plot(t, sigma0, color=c_mstar, lw=2)
    # ax[0].set(
    #     ylabel=r"$\Sigma_{\rm ISM} ~ [M_\odot \, kpc^{-2}]$", yscale="log"
    # )

    # ax[1].plot(t, virial_rad, color=c_mhalo, lw=2, label=r"$R_{\rm vir}$")
    # ax[1].plot(t, disk_radius, color=c_mstar, lw=2, label=r"$R_{\rm disk}$")
    # ax[1].axhline(1.0, color="k", lw=1, ls="--")
    # ax[1].set(ylabel=r"[kpc]", yscale="log",  xlabel=r"${\rm time\: [Gyr]}$")
    # ax[1].legend()
    # plt.show()

    # ############################################################
    ################### top panels, plot the mass and energy evolution
    ############################################################
    fig, ax = plt.subplots(2, 2, figsize=(8, 5.5), dpi=300, sharex="row")

    fig.subplots_adjust(hspace=0.15)
    ax = ax.flatten()
    ax[0].plot(t, mbh, label=r"$M_{\rm \bullet}$", color=c_bh, lw=2)
    ax[0].plot(
        t, mhalo_t * f_baryon, label=r"$f_{\rm b} M_{\rm halo}$", color=c_mhalo, lw=2
    )
    ax[0].plot(t, mgas_t, label=r"$M_{\rm ISM}$", color=c_mism, lw=2)
    ax[0].plot(t, mcgm_t, label=r"$M_{\rm CGM}$", color=c_mcgm, lw=2)
    ax[0].plot(t, mcgm_hot_t, label=r"$M_{\rm CGM, hot}$", color=c_mcgm_hot, lw=2)
    ax[0].plot(
        t, mcgm_cold_t, label=r"$M_{\rm CGM, cold}$", color=c_mcgm_cold, lw=2, ls="--"
    )
    
    ax[0].plot(t, mstar_t, label=r"$M_{\star}$", color=c_mstar, lw=2)
    ax[0].plot(
        t, mbulge_eff, label=r"$M_{\rm bulge}$", color="tab:green", ls="--", lw=2
    )

    # the energy evolution
    ax[1].plot(t, egy_bh, label=r"$E_{\rm \bullet, therm}$", color=c_bh, lw=2)
    ax[1].plot(t, egy_eject_t, label=r"$E_{\rm CGM, out}$", color=c_mhalo, lw=2)
    ax[1].plot(t, egy_t, label=r"$E_{\rm ISM, winds}$", color=c_mstar, lw=2)
    ax[1].plot(t, egy_cgm_t, label=r"$E_{\rm CGM}$", color=c_mcgm, lw=2)
    ax[1].plot(
        t, egy_accrete_t, label=r"$E_{\rm CGM, accrete}$", color=c_mcgm_hot, lw=2
    )
    ax[1].plot(
        t, egy_radloss_t, label=r"$E_{\rm CGM, cooling}$", color=c_mcgm_cold, lw=2
    )

    # and the rates of change of masses
    ax[2].plot(
    t, dot_m_bh, label=r"$\dot{M}_{\rm \bullet}$", color=c_bh, lw=2
    )
    ax[2].plot(t, dot_m_sfr, label=r"$\dot{M}_\star$", color=c_mstar, lw=2)
    ax[2].plot(t, dot_m_ism_wind, label=r"$\eta_M \dot{M}_\star$", color=c_mism, lw=2)
    ax[2].plot(
        t, dot_m_cgm_in, label=r"$\dot{M}_{\rm CGM, infall}$", color=c_mhalo, lw=2
    )
    ax[2].plot(t, dot_m_cgm_out, label=r"$\dot{M}_{\rm CGM, out}$", color=c_mcgm, lw=2)
    ax[2].plot(
        t,
        dot_m_cgm_hot,
        label=r"$\dot{M}_{\rm CGM, hot}$",
        color=c_mcgm_hot,
        lw=2,
        alpha=0.7,
    )
    ax[2].plot(
        t,
        dot_m_cgm_cold,
        label=r"$\dot{M}_{\rm CGM, cold}$",
        color=c_mcgm_cold,
        lw=2,
        alpha=0.7,
        ls="--",
    )
    # ax[2].plot(
    #     t, dot_m_bulge, label=r"$\dot{M}_{\rm bulge}$", color=c_mstar, ls="--", lw=2
    # )
    
    # as well as the energy rates
    ax[3].plot(
        t,
        dot_e_bh_thermfeedback,
        label=r"$\dot{E}_{\rm \bullet, therm}$",
        color=c_bh,
        lw=2,
    )
    ax[3].plot(t, dot_egy_cgm, label=r"$\dot{E}_{\rm CGM}$", color=c_mcgm, lw=2)
    ax[3].plot(
        t, dot_e_cgm_out, label=r"$\dot{E}_{\rm CGM, eject}$", color=c_mhalo, lw=2
    )
    ax[3].plot(
        t, dot_e_ism_wind, label=r"$\dot{E}_{\rm ISM, winds}$", color=c_mstar, lw=2
    )
    ax[3].plot(
        t,
        dot_e_cgm_cooling,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
        color=c_mcgm_cold,
        lw=2,
        alpha=0.7,
    )
    ax[3].plot(
        t,
        dot_e_cgm_in,
        label=r"$\dot{E}_{\rm CGM, accrete}$",
        color=c_mcgm_hot,
        lw=2,
        alpha=0.7,
    )

    # set axes labels
    ax[0].set(
        ylabel=r"$\rm Masses ~ [{\rm M_{\odot}}]$",
        yscale="log",
        ylim=(1e4, (mhalo_t * f_baryon)[-1]),
    )
    ax[0].legend(
        ncols=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(-0.2, 1.15),
        loc="lower left",
    )
    max_y = max([line.get_ydata().max() for line in ax[1].lines])
    ax[1].set(
        ylabel=r"$\rm Total ~Energy ~ [{\rm erg}]$",
        yscale="log",
        ylim=(1e51, max_y * 1.5),
    )
    ax[1].legend(
        ncols=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(-0.05, 1.15),
        loc="lower left",
    )
    max_y = max([line.get_ydata().max() for line in ax[2].lines])
    ax[2].set(
        ylabel=r"$\rm Mass ~ Rates ~  [{\rm M_{\odot} \ Gyr^{-1}}]$",
        yscale="log",
        ylim=(dot_m_cgm_in.max() / 1e6, max_y * 1.5),
    )
    ax[2].legend(
        ncols=4,
        frameon=False,
        loc="upper left",
        fontsize=8,
        bbox_to_anchor=(-0.2,  -0.05),
    )
    max_y = max([line.get_ydata().max() for line in ax[3].lines])
    ax[3].set(
        ylabel=r"$\rm Energy ~ Rates ~  [{\rm erg \ Gyr^{-1}}]$",
        yscale="log",
        ylim=(1e51, max_y * 1.5),
    )
    ax[3].legend(
        ncols=2, frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, -0.05)
    )

    # add a time label to the bottom of the second row
    fig.text(0.5, -0.01, r"${\rm time\: [Gyr]}$", ha="center", fontsize=10)
    ## set all the axis x limmits to the same range

    # # add a twin axis for the redshift
    for a in ax:
        a.set_xlim(t[0] * 0.99, t[-1])

    # make a twin redshift axis for the top row, using z
    # get the current x axis labels of the first row and their
    x_axis_tick_labels = ax[0].get_xticks()
    x_axis_tick_labels = np.array(x_axis_tick_labels)[2:-1]
    # prepend the lowest time value to the beginning of the array
    x_axis_tick_labels = np.insert(x_axis_tick_labels, 0, t[0])
    for i in range(2):
        ax2 = ax[i].twiny()
        # make sure the ranges are the same
        ax2.plot(t, dot_e_cgm_out, color="k", alpha=0)
        ax2.set_xlim(t[0] * 0.99, t[-1])
        t_ticks = x_axis_tick_labels  # [8,  5,4, 3, 2, 1,  0.001]
        # z_ticks = np.geomspace(np.max(adaptive_z).value, np.min(adaptive_z).value, 5)
        # t_ticks = LCDM.age(z_ticks).value  # Convert redshift to corresponding time
        z_ticks = cosmology.z_at_value(LCDM.age, t_ticks * u.Gyr).value
        ax2.set_xticks(x_axis_tick_labels)
        ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
        ax2.set_xlim(t[0] * 0.99, t[-1])
        ax2.set_xlabel(r"$z$")
        # remove minor ticks in the top x axis
        ax2.minorticks_off()
        ax[i].minorticks_on()
    # put text on the corner left
    # ax[0].text(
    #     0,
    #     1.2,
    #     run_text,
    #     transform=ax[0].transAxes,
    #     fontsize=10,
    #     verticalalignment="bottom",
    #     horizontalalignment="left",
    # )

    for axes in ax:
        for line in axes.lines:
            line.set_zorder(1)

    # plt.savefig("./figures/mwmass_rates.png", dpi=200,          bbox_inches="tight",
    #     pad_inches=0.05)
    # plt.show()

    # return t, t_dep_eff, f_prevent[-1]
    return ax


def accretion_rates(results, derived_quant, title: str):
    t = derived_quant["sim_time"]
    z_red = derived_quant["z"]
    tvir = derived_quant["tvir"]
    virial_rad = derived_quant["halo_rvir"]
    disk_radius = 0.02 * virial_rad  # assume disk radius is 10% of the virial radius

    dot_e_cgm_out = derived_quant["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_quant["dot_e_cgm_cooling"]
    dot_e_cgm_in = derived_quant["dot_e_cgm_in"]
    dot_e_ism_wind = derived_quant["dot_e_ism_wind"]
    dot_m_cgm_out = derived_quant["dot_m_cgm_out"]
    dot_m_cgm_hot = derived_quant["dot_m_cgm_hot"]  # hot gas cooling to cold gas
    dot_m_cgm_cold = derived_quant["dot_m_cgm_cold"]  # cold gas falling into ISM
    dot_m_cgm_in = derived_quant["dot_m_cgm_in"]
    dot_m_ism_wind = derived_quant["dot_m_ism_wind"]
    dot_m_sfr = derived_quant["dot_m_sfr"]
    dot_m_bulge = derived_quant["dot_m_bulge"]
    cooling_lambda = derived_quant["cooling_lambda"]
    f_prevent = derived_quant["f_prevent"]
    f_star = derived_quant["f_star"]

    t_depletion = derived_quant["t_dep"]

    t_dynamical = derived_quant["t_dyn"]
    tcool_real = derived_quant["tcool_real"]  # purely cooling time based on CGM
    tcool_eff = derived_quant["tcool_eff"]  # added some fraction of dynamical time
    tcool_grav = derived_quant["tcool_grav"]  # added dynamical time
    t_ejection = derived_quant["t_ejection"]

    adaptive_z = results["z"]
    t_adaptive = results["t"]
    mgas_t = results["m_ism"]
    mstar_t = results["m_star"]
    mass_bulge_t = results["m_bulge"]
    disk_stellar_mass = mstar_t - mass_bulge_t
    bulge_disk_ratio = mass_bulge_t / mstar_t
    mcgm_t = results["m_cgm"]
    mcgm_hot_t = results["m_cgm_hot"]
    mcgm_cold_t = results["m_cgm_cold"]
    mmetals_t = results["m_metals"]
    mhalo_t = results["m_halo"]
    egy_t = results["egy_ism_wind"]
    egy_radloss_t = results["egy_radloss"]
    egy_eject_t = results["egy_eject"]
    egy_accrete_t = results["egy_accrete"]
    egy_cgm_t = results["egy_cgm"]
    metal_cgm_mass = results["metal_cgm_mass"]
    metal_cgm_mass_sol = results["metal_cgm_mass_sol"]
    sigma0 = mgas_t / (np.pi * disk_radius**2)  # surface density of the ISM gas

    mass_of_cgm = mcgm_t * u.solMass
    egy_cgm = egy_cgm_t * u.erg

    mu = 0.6 * consts.m_p
    kb = consts.k_B

    T_eff = ((egy_cgm / mass_of_cgm) * (mu / kb)).to(u.K)
    c_sound = c_s = np.sqrt(kb * T_eff / mu).to(u.km / u.s)
    rho0 = density0(mCGM=mcgm_hot_t, r0=virial_rad * 0.1, Rvir=virial_rad, alpha=1.4)

    # plot the quantities going into the bondi accretion rates

    fig, ax = plt.subplots(3, 1, figsize=(5, 7), dpi=200)
    ax[0].plot(t, T_eff)
    ax[1].plot(t, c_sound)
    ax[2].plot(t, rho0)

    ax[0].set(yscale="log", ylabel=r"$T_{\rm CGM}$")
    ax[1].set(yscale="log", ylabel=r"$c_s$ [km/s]")
    ax[2].set(
        yscale="log", xlabel="t [Gyr]", ylabel=r"$\rho_{\rm 0, hot}$ [Msun/kpc^3]"
    )
    plt.show()


# mhalo_z0 = 1e12 * u.Msun
# t_span = (0.1, 13)  # gyrs
# eta_m = 0.1
# eta_e = 0.1
# eta_z = 0.2
# ##fmt: off
# model = CGM_regulator(
#     mhalo_z0,
#     t_span,

#     eta_z=eta_z,
#     cooling_dynamic_time_norm=1,
#     disk_scale_length=0.02,
#     ep_bh_feedback_eff=0.02,
#     bondi_boost=1e4,
    
# )
# # ## fmt: on
# run = model.run_halo()
# results = model.get_results()
# derived = model.get_derived_quantities()
# ax = halo_diagnostics_v2(results, derived, title="latest, 2 phase")

# # accretion_rates(
# #     results, derived, title="latest, 2 phase"
# # )
# # halo_timescales(results, derived, title="latest, 2 phase")
# #%%
# # try debugging different normalizations for the accretion energy 
# accretion_norm = np.geomspace(0.001,0.1,20)
# bondi_boosts = np.geomspace(100, 1e5, 20)
# mhalo_z0 = 1e12 * u.Msun
# t_span = (0.1, 13)  # gyrs

# ##fmt: off

# for boost in bondi_boosts:
#     model = CGM_regulator(
#         mhalo_z0,
#         t_span,
#         cooling_dynamic_time_norm=1,
#         disk_scale_length=0.02,
#         ep_bh_feedback_eff=0.02,
#         bondi_boost=boost
#     )
#     # fmt: on
#     run = model.run_halo()
#     results = model.get_results()
#     derived = model.get_derived_quantities()

#     ax = halo_diagnostics_v2(results, derived, title="latest, 2 phase")
#     # add a figure text for the value of dbug_norm_for_accretion_energy_rate used
  
#     ax[0].text(0.05, 1.5, f"bondi boost= {boost:.3f}", transform=ax[0].transAxes)

#     # make the figure naming by replacing dot with p
#     name = f"{int(boost):05d}"
    
#     plt.savefig(f"./figures/bondi_boost_test/boost_{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
#     plt.show()

#     # accretion_rates(
#     #     results, derived, title="latest, 2 phase"
#     # )
#     # halo_timescales(results, derived, title="latest, 2 phase")
    
#     ######  more detailed breakdown of accretion
#     m_bulge = results["m_bulge"]
#     dot_e_bh_thermfeedback = derived["dot_e_bh_thermfeedback"]
#     dot_m_bh_bondi = derived["dot_m_bh_bondi"]  # has units of msun/gyr
#     dot_m_bh_tla = derived["dot_m_bh_tla"]  # has units of msun/gyr
#     m_gas_in_bh_disk = derived["m_gas_in_bh_disk"]
#     m_star_in_bh_disk = derived["m_star_in_bh_disk"]
#     m_bh_disk = m_gas_in_bh_disk + m_star_in_bh_disk
#     bh_mass = results["m_bh"] * u.Msun
#     eps_r = 0.1
#     mdot_edd = (
#             4 * np.pi * consts.G * bh_mass* consts.m_p
#         ) / (eps_r * consts.sigma_T * consts.c)
#     mdot_edd = mdot_edd.to(u.Msun/u.Gyr)
#     effective_bulge_mass = np.copy(m_bulge)
#     mask = effective_bulge_mass < np.pi * 1e9
#     effective_bulge_mass[mask] = 0
#     effective_bulge_mass[~mask] = effective_bulge_mass [~mask] - (np.pi *1e9)


#     t = results["t"]
#     fig, ax = plt.subplots(4, 2, figsize=(9, 8), dpi=180)
#     ax=ax.T
#     ax = ax.flatten()
#     ax[0].plot(t, m_bulge , label="SF within 1 kpc")
#     ax[0].plot(t, effective_bulge_mass, label=r"max[M ( <1kpc) - $\pi \times$ 1e9 Msun, 0 ]")
#     ax[0].plot(t,  m_bulge - effective_bulge_mass, label="BH kernel stellar disk component", ls= "--")

#     # plot all the accretion rates
#     ax[1].plot(t, dot_m_bh_bondi, label=r"bondi")
#     ax[1].plot(t, dot_m_bh_tla, label=r"tla")
#     ax[1].plot(t, dot_m_bh_tla + dot_m_bh_bondi, label="total", ls="--")

#     # now do it in terms of eddington ratio
#     ax[2].plot(t, np.log10(dot_m_bh_bondi / mdot_edd.value), label=r"bondi")
#     ax[2].plot(t, np.log10(dot_m_bh_tla / mdot_edd.value), label=r"tla")
#     ax[2].plot(t, np.log10((dot_m_bh_tla + dot_m_bh_bondi) / mdot_edd.value), label="total", ls="--")

#     ax[3].plot(t, bh_mass, label=r"bh mass")
#     ax[3].plot(t, m_gas_in_bh_disk, label=r"BH gas mass in disk")
#     ax[3].plot(t, m_star_in_bh_disk, label=r"BH star mass in disk")
#     ax[3].plot(t,  m_gas_in_bh_disk + m_star_in_bh_disk + effective_bulge_mass, label="enclosed mass")
#     ax[3].plot(t,  m_bh_disk, label="BH disk mass", ls="--")

#     ax[0].set(yscale="log", ylabel="msun")
#     ax[0].legend()
#     ax[1].set(yscale="log", ylabel= "msun / gyr")
#     ax[1].legend()
#     ax[2].set(ylabel= "log eddington ratio")
#     ax[2].legend()
#     ax[3].set(yscale="log", ylabel= "msun")
#     ax[3].legend(ncol=2, fontsize=8)

#     # now, plot the derived quantities f_0, f_gas, f_disk

#     # Calculate derived quantities for plotting
#     f_disk = m_bh_disk / (m_gas_in_bh_disk + m_star_in_bh_disk + effective_bulge_mass)
#     f_gas = m_gas_in_bh_disk / m_bh_disk
#     f_0 = 0.31 * f_disk**2 * (m_bh_disk / 1e9) ** -0.666

#     ax[4].plot(t, f_disk, label="f_disk")
#     ax[4].set(ylabel=r"$f_{\rm disk}$")
#     ax[5].plot(t, f_gas, label="f_gas")
#     ax[5].set(ylabel=r"$f_{\rm gas}$", yscale="log")
#     ax[6].plot(t, f_0, label=r"$f_0$")
#     ax[6].set(ylabel=r"$f_0$", yscale="log")

#     f_term = (1 + (f_0 / f_gas)) **-1
#     ax[7].plot(t, f_term,)
#     ax[7].set(ylabel=r"$(1 + f_0 / f_{\rm gas})^{-1}$", yscale="log")
    
#     ax[0].text(0.05, 1.5, f"bondi boost= {boost:.3f}", transform=ax[0].transAxes)
    
#     plt.savefig(f"./figures/bondi_boost_breakdown/boost_breakdown_{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
  
#     plt.show()





# # %% let's do a scan through of possible values for BH feedback normalization

# epsilon_feedback = np.geomspace(0.005, 0.03, 5)
# fig, ax = plt.subplots(7, 1, figsize=(7, 12), dpi=180)
# for i, eps in enumerate(epsilon_feedback):
        
#     model = CGM_regulator(
#         1e13 * u.Msun,
#         t_span,
#         eta_m=eta_m,
#         eta_e=eta_e,
#         eta_z=eta_z,
#         cooling_dynamic_time_norm=1,
#         disk_scale_length=0.02,
#         ep_bh_feedback_eff=eps,
#     )
#     # fmt: on
#     run = model.run_halo()
#     results = model.get_results()
#     derived = model.get_derived_quantities()
#     t = results["t"]
#     m_bulge = results["m_bulge"]
#     m_star = results["m_star"]
#     dot_e_bh_thermfeedback = derived["dot_e_bh_thermfeedback"] 
#     dot_m_bh_bondi = derived["dot_m_bh_bondi"] # has units of msun/gyr
#     dot_m_bh_tla = derived["dot_m_bh_tla"]  # has units of msun/gyr
#     m_gas_in_bh_disk = derived["m_gas_in_bh_disk"]
#     m_star_in_bh_disk = derived["m_star_in_bh_disk"]
#     m_bh_disk = m_gas_in_bh_disk + m_star_in_bh_disk
#     bh_mass = results["m_bh"] * u.Msun
#     eps_r = 0.1
#     mdot_edd = (
#             4 * np.pi * consts.G * bh_mass* consts.m_p
#         ) / (eps_r * consts.sigma_T * consts.c)
#     mdot_edd = mdot_edd.to(u.Msun/u.Gyr)
#     effective_bulge_mass = np.copy(m_bulge)
#     mask = effective_bulge_mass < np.pi * 1e9
#     effective_bulge_mass[mask] = 0
#     effective_bulge_mass[~mask] = effective_bulge_mass [~mask] - (np.pi *1e9)
#     ax[0].plot(t, m_star, label=r"$\epsilon_{{\rm feedback}} = {:.4f}$".format(eps))
#     ax[0].set(yscale="log", ylabel= r"$M_\star$", ylim=(1e8, 1e11))
#     ax[0].legend(loc="lower center", fontsize=8, ncol=4, bbox_to_anchor=(0.5, 1.2))

#     ax[1].plot(t, effective_bulge_mass)
#     ax[1].set(yscale="log", ylabel= r"$M_{\rm bulge}$")
    
#     # then see how the TLA quantities are
#     f_disk = m_bh_disk / (m_gas_in_bh_disk + m_star_in_bh_disk + effective_bulge_mass)
#     f_gas = m_gas_in_bh_disk / m_bh_disk
#     f_0 = 0.31 * f_disk**2 * (m_bh_disk / 1e9) ** -0.666
#     ax[2].plot(t, f_disk)
    
#     ax[2].set(ylabel=r"$f_{\rm disk}$")
#     ax[3].plot(t, f_gas)
#     ax[3].set(ylabel=r"$f_{\rm gas}$", yscale="log")
#     ax[4].plot(t, f_0)
#     ax[4].set(ylabel=r"$f_0$", yscale="log")
#     f_term = (1 + (f_0 / f_gas)) **-1
#     ax[5].plot(t, f_term)
#     ax[5].set(ylabel=r"$(1 + f_0 / f_{\rm gas})^{-1}$", yscale="log")
    
#     # add the BH mass
#     ax[6].plot(t, bh_mass, label=r"$M_{\rm \bullet}$")
#     ax[6].set(yscale="log", ylabel=r"$M_{\rm \bullet} ~ [{\rm M_\odot}]$", ylim=(1e5,1e7))
#     ax[6].set_xlabel(r"${\rm time\: [Gyr]}$")
    
    
# plt.show()
    
   


# %% see how the star formation mass differs between the two models as a function of time
# fig, ax = plt.subplots(2, 1, figsize=(5, 4), dpi=300)
# ax[0].plot(tsim, m_star, label="2 phase",  lw=2)
# ax[0].plot(tsim_base, m_star_base, label="Baseline model", lw=2)
# ax[0].plot(old_tsim, old_m_star, label="2 phase (no KS)")
# ax[0].set(
#     ylabel=r"$M_\star ~ [{\rm M_\odot}]$",
#     # yscale="log"
# )

# ax[1].plot(tsim, dot_m_sfr)
# ax[1].plot(tsim_base, dot_m_sfr_base)
# ax[1].plot(old_tsim, old_dot_m_sfr)
# ax[1].set(
#     ylabel=r"$\dot{M}_\star ~ [{\rm M_\odot \ Gyr^{-1}}]$",
#     yscale="log",
#     # ylim=(1e4, 1e10),
#     xlabel=r"${\rm time\: [Gyr]}$"
# )

# ax[0].legend(title=r"$M_{{\rm halo}} = {:.2e}$ Msun".format(mhalo_z0.value))
# plt.show()


# %%
# f_prevents = []
# halo_masses = np.geomspace(1e11, 1e13, 5) * u.Msun
# mstar = []
# mbulge = []
# fig, ax = plt.subplots(4, 1, figsize=(5, 8), dpi=300, sharex=True)
# plt.subplots_adjust(hspace=0.0)
# for  i, mhalo_z0 in enumerate(halo_masses):
#     model = CGM_regulator(
#         mhalo_z0,
#         t_span,
#         eta_m=eta_m,
#         eta_e=eta_e,
#         eta_z=eta_z,
#         cooling_dynamic_time_norm=1,
#         disk_scale_length=0.02,
#     )
#     run = model.run_halo()
#     results = model.get_results()
#     derived = model.get_derived_quantities()
#     # get the stellar mass and the bulge mass
#     t = results["t"]
#     m_ism_t = results["m_ism"]
#     mstar_t = results["m_star"]
#     mass_bulge_t = results["m_bulge"]
#     virial_rad = derived["halo_rvir"]
#     disk_radius = 0.02 * virial_rad
#     sigma0 = m_ism_t / (np.pi * disk_radius**2)  # surface density of the ISM gas
#     disk_radius = 0.02 * virial_rad  # assume disk radius is 10% of the virial radius


#     ax[0].plot(t, sigma0, lw=2, label = "Mhalo = {:.2e} Msun".format(mhalo_z0.value), alpha=0.7)
#     ax[0].legend(bbox_to_anchor=(0, 1.2), loc="lower left", ncol=2, frameon=False)
#     # get the current color in the cycler
#     c = ax[0].lines[-1].get_color()
#     ax[0].set(
#         ylabel=r"$\Sigma_{\rm ISM} ~ [M_\odot \, kpc^{-2}]$", yscale="log"
#     )
#     if i ==0:
#         ax[1].plot(t, virial_rad, lw=2, label=r"$R_{\rm vir}$", ls = "-", color=c)
#         ax[1].plot(t, disk_radius,  lw=2, label=r"$R_{\rm disk}$", ls = "--", color=c)
#     else:
#         ax[1].plot(t, virial_rad, lw=2,  ls = "-", color=c)
#         ax[1].plot(t, disk_radius,  lw=2,  ls = "--", color=c)


#     ax[1].axhline(1.0, color="k", lw=1, )
#     ax[1].set(ylabel=r"[kpc]", yscale="log")

#     # plot the stellar mass and the bulge mass
#     ax[2].plot(t, mstar_t, lw=2,  color=c, alpha=0.5)
#     ax[2].plot(t, mass_bulge_t, lw=2, ls=":", color=c)
#     ax[2].set(yscale="log", ylabel="stellar mass, dotted in bulge")

#     #plot ratio of bulge to stellar mass
#     ax[3].plot(t, mass_bulge_t / mstar_t, lw=2, ls=":", color=c, alpha=0.5)
#     ax[3].set(
#         xlabel=r"${\rm time\: [Gyr]}$",
#         ylabel=r"$M_{\rm bulge} / M_\star$",
#         yscale="log")
# ax[1].legend()
# plt.show()

# t, t_dep_eff, f_p = halo_diagnostics_v2(
# results,
# derived,
# title="latest, 2 phase with f_prevent in Ecgm in",
# )
# f_prevents.append(f_p)


# %%
# fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
# ax.scatter(halo_masses, f_prevents, label=r" value of $f_{\rm prevent}$ at z = 0", color="tab:blue")
# ax.set(
#     xlabel=r"$M_{\rm halo} ~ [{\rm M_\odot}]$",
#     ylabel=r"$f_{\rm prevent}$",
#     xscale="log",
#     yscale="linear",
# )
# ax.legend()
# plt.show()
# %%


# # halo_masses = results["m_halo"]
# time = results["t"]
# halo_rvir = derived["halo_rvir"]
# z = derived["z"]
# t = derived["sim_time"]
# halo_sfe = derived["f_star"][-1]


# rdisk_t_dep_eff.append(t_dep_eff)
# t_test.append(t)

# ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
# for r, rdisk_val in enumerate(r_disk_try ):
#     ax.plot(t_test[r], rdisk_t_dep_eff[r], label=r"$r_{{\rm disk}} = {:.3f} ~ {{\rm kpc}}$".format(rdisk_val))
# ax.set(
#     xlabel=r"$t ~ [{\rm Gyr}]$",
#     ylabel=r"$t_{\rm dep, eff} ~ [{\rm Gyr}]$",
#     yscale="log",
#     xscale="log",
#     ylim=(1e-3, 10),
# )
# ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, frameon=False)

# plt.show()

# %%
# plot_halo_profile(results, derived)
# plot_halo_diagnostics(
#     results,
#     derived,
#     title="updating mass and energy loadings",
# )


# #%%

# alpha_m = 0.3
# alpha_e = 0.1

# halo_vcirc = circular_velocity(halo_masses * u.Msun, halo_rvir* u.kpc)
# old_mass_loading = custom_mass_loading(halo_masses, A=10, alpha=-1.4)
# old_energy_loading = custom_energy_loading(halo_masses, A=0.1, alpha=-0.5)

# new_mass_loading = vcirc_mass_loading(halo_vcirc, alpha_m=alpha_m)
# new_energy_loading = vcirc_energy_loading(halo_vcirc, alpha_e=alpha_e )


# fig, ax = plt.subplots(2, 1, figsize=(6, 5), dpi=300, sharex=True)
# plt.subplots_adjust(hspace=0)
# ax[0].plot(time, old_mass_loading, label="mass based", color="k")
# ax[0].plot(time, new_mass_loading,  color="tab:blue",label=r"halo circular velocity based $\alpha_M = {:}, \alpha_E= {:}$".format(alpha_m, alpha_e))
# ax[0].set(

#     ylabel=r"$\eta_{\rm M}$",
#     yscale="log"
# )
# ax[0].legend(ncols=1, title="{:.2e}".format(mhalo_z0))

# ax[1].plot(time, old_energy_loading, label="mass based", color="k")
# ax[1].plot(time, new_energy_loading, label="vcirc based", color="tab:blue")
# ax[1].set(
#     xlabel="t [Gyr]",
#     ylabel=r"$\eta_{\rm E}$",
#     yscale="log"
# )
# # add a twin axis for the redshift
# ax2 = ax[0].twiny()
# ax2.plot(time, old_energy_loading, color="k", alpha=0)
# tick_times = np.array([0.10, 1, 2, 4, 5, 6, 8, 12,13])  # Gyr
# z_ticks = cosmology.z_at_value(LCDM.age, tick_times * u.Gyr).value

# # Apply to twin axis
# ax2.set_xticks(tick_times)
# ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
# ax2.set_xlabel(r"$z$")

# plt.show()


# %% run a grid of model for the depletion time test model
# dep_time_norm_vals = np.linspace(0.1, 0.9, 5)
# sfes = []
# # plot sfe vs depletion time
# for td in dep_time_norm_vals:
#     model = CGM_regulator(
#         mhalo_z0, t_span, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z, dep_time_norm=td
#     )
#     run = model.run_halo()
#     results = model.get_results()
#     derived = model.get_derived_quantities()
#     halo_sfe = derived["f_star"][-1]
#     # plot_halo_profile(results, derived, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z)
#     plot_halo_diagnostics(results, derived, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z)
#     sfes.append(halo_sfe)

# fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
# ax.plot(dep_time_norm_vals, sfes, marker="o", color="k")
# ax.set(xlabel=r"$\epsilon_{\rm dep}$", ylabel=r"$f_{\star}$")
# plt.show()

# %% run a grid of models to test the effect of the cooling_dynamic_time_norm

# dynamic_time_normvals = np.geomspace(0.01, 1, 10)
# sfes = []
# # plot sfe vs depletion time
# for td in dynamic_time_normvals:
#     model = CGM_regulator(
#         mhalo_z0,
#         t_span,
#         eta_m=eta_m,
#         eta_e=eta_e,
#         eta_z=eta_z,
#         dep_time_norm=0.4,
#         cooling_dynamic_time_norm=td,
#     )
#     run = model.run_halo()
#     results = model.get_results()
#     derived = model.get_derived_quantities()
#     halo_sfe = derived["f_star"][-1]
#     # plot_halo_profile(results, derived, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z)
#     # plot_halo_diagnostics(results, derived)
#     sfes.append(halo_sfe)

# fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
# ax.plot(dynamic_time_normvals, sfes, marker="o", color="k")
# ax.set(xlabel=r"norm to dynamical time", ylabel=r"$f_{\star}$", xscale="log")
# plt.show()


# %% plot sfe vs mhalo
# mhs = np.geomspace(1e10, 1e14, 8) * u.Msun
# sfes = []
# for mh in mhs:
#     model = CGM_regulator(mh, t_span, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z)
#     run = model.run_halo()
#     results = model.get_results()
#     derived = model.get_derived_quantities()
#     halo_sfe = derived["f_star"][-1]
#     plot_halo_profile(results, derived, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z)
#     plot_halo_diagnostics(results, derived, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z)
#     sfes.append(halo_sfe)
# fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
# ax.plot(mhs.value, sfes, marker="o", color="k")
# ax.set(xlabel=r"$M_{\rm halo}$ [Msun]", ylabel=r"$f_{\star}$")
# plt.xscale("log")
# plt.show()


# %%


# %% vary the eta_m
# def vary_mass_loading(z_obs, eta_m, eta_e=0.1, eta_z=0.3):
#     mhalos = np.geomspace(1e10, 1e12, 10) * u.Msun
#     t_init = 0.1  # Gyr
#     t_final = LCDM.age(z_obs).value  # Gyr
#     t_span = (t_init, t_final)  # span of the integration
#     mhalos_at_z_obs = []
#     x = []
#     y = []

#     for mass_loading_vals in eta_m:
#         mhalo_obs = []
#         mstar_obs = []
#         for midx, mhalo in enumerate(mhalos):
#             mhalo_z0 = mhalo_at_z0(mhalo, z_obs)
#             print("observing halo with mass {:.2e} at z = {:.2f}".format(mhalo, z_obs))
#             print("mass of halo at z=0 is {:.2e}".format(mhalo_z0))
#             gridmodel = CGM_regulator(
#                 mhalo_z0 * u.Msun,
#                 t_span,
#                 tstep=0.001,
#                 eta_m=mass_loading_vals,
#                 eta_e=eta_e,
#                 eta_z=eta_z,
#             )
#             run = gridmodel.run_halo()
#             results = gridmodel.get_results()
#             derived = gridmodel.get_derived_quantities()
#             m_star = results["m_star"][-1]
#             m_halo = results["m_halo"][-1]
#             mhalo_obs.append(m_halo)
#             mstar_obs.append(m_star)

#         mhalo_obs = np.array(mhalo_obs)
#         mstar_obs = np.array(mstar_obs)
#         x.append(mhalo_obs)
#         y.append(mstar_obs)

#     return x, y


# %%


# # we want to observe these halos at a given redshift
# z_obs = 0.01  # redshift of observation
# t_obs = LCDM.age(z_obs).value  # age of the universe at this redshift

# t_initial = 1  # Gyr, start the integration at this times
# t_span = (t_initial, t_obs)  #  span of the integration
# print(t_span)
# z_init = cosmology.z_at_value(LCDM.age, t_initial * u.Gyr)
# # what would be the masses of these halos at t_initial
# mhalos_init = [initial_mhalo(mhalo_zobs, t_span) for mhalo_zobs in mhalos_zobs]
# # what would be the masses of these halos at z=0 and at t_inital
# # mhalos_z0 = [mhalo_at_z0(mhalo_zobs, z_obs) for mhalo_zobs in mhalos_zobs]
# 1e12
# mhalos_init = np.array(mhalos_init) * u.Msun
# # mhalos_z0 = np.array(mhalos_z0) * u.Msun

# mass_loading_constant = 0.5
# eta_z = 0.3
# # each row has all the masses for a given energy loading
# mstar_etam_const = []
# mgas_etam_const = []

# start_time = time.time()

# for i, energy_loading in enumerate(tqdm(energy_loading_grid)):
#     # for each energy loading value, we run the models for all the halos
#     galaxy_stellar_masses = []
#     galaxy_gas_masses = []

#     for j, mhalo_t0 in enumerate(tqdm(mhalos_init)):
#         eta_m = mass_loading_constant
#         eta_e = energy_loading
#         # eta_m =  20 * (mhalos_zobs[i] / (1e10 * u.solMass)) ** -0.7
#         # eta_e = 0.15 * (mhalos_zobs[j] / (1e12 * u.solMass)) ** -0.5
#         print("> evolving halo with mass {:.2e} at z = {:.2f}".format(mhalo_t0, z_init))
#         print(
#             "> to redshift {:.2f} where the halo has mass {:.2e}".format(
#                 z_obs, mhalos_zobs[j]
#             )
#         )
#         print("> eta_m = {:.2f}, eta_e = {:.2f}".format(eta_m, eta_e))
# 1e12
#         model = CGM_regulator(
#             mhalo_t0, t_span, tstep=0.01, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z
#         )
#         run = model.run_halo()
#         results = model.get_results()
#         galaxy_stellar_masses.append(results["m_star"][-1])
#         galaxy_gas_masses.append(results["m_gas"][-1])

#     mstar_etam_const.append(galaxy_stellar_masses)
#     mgas_etam_const.append(galaxy_gas_masses)

# end_time = time.time()
# print(f"Total time taken: {end_time - start_time} seconds")

# # %% do the 1Param variation
# fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
# ax.plot(10**loghm, 10**logSMHM, color="k", ls="--")
# ax.fill_between(
#     10**loghm,
#     10 ** (logSMHM + SMHMerr[1, :]),
#     10 ** (logSMHM - SMHMerr[0, :]),
#     color="gray",
#     alpha=0.6,
#     label="Behroozi et al. 2019",
# )
# ax.set(
#     xlabel=r"M$_{\rm halo}$ [M$_\odot$]",
#     ylabel=r"M$_\star$/M$_{\rm halo}$ [M$_\odot$]",
#     yscale="log",
#     xscale="log",
# )
# # lessthan_mw_mask = mhalos_z0.value < 1e12
# for i, energy_loading in enumerate(energy_loading_grid):
#     smhm = np.array(mstar_etam_const[i]) / mhalos_zobs.value
#     ax.plot(mhalos_zobs, smhm, label=r"$\eta_e = {:.1f}$".format(energy_loading))
# ax.set(
#     xscale="log",
#     yscale="log",
#     xlabel=r"$M_{\rm halo}$",
#     ylabel=r"$M_{\star} / M_{\rm halo}$",
# )
# title = r"$\eta_M = {:.1f}, z_{{\rm obs}} = {:.1f}$".format(
#     mass_loading_constant, z_obs
# )
# # text top left corner, saying what halo infall mode
# ax.text(0.05, 0.95, "Halo infall mode: Dekkel", transform=ax.transAxes, fontsize=10)
# # title= r"$\eta_M$ = 20 (mhalo at z obs) / 1e10 Msun, $z_{{\rm obs}} = {:}$".format(z_obs)
# ax.legend(title=title)
# plt.show()

# ##########################################################################################
# # %% now vary eta_m and eta_e together, we want to observe these halos at a given redshift1e12ce(0.1, 1, 5)

# z_array = [6, 5, 4, 1, 0.001]
# # z_array = np.geomspace(10, 0.0001, 15)
# galaxy_stellar_masses_zevol = []
# galaxy_gas_masses_zevol = []
# halo_masses_zevol = []
# f_stars = []
# for z_obs in z_array:

#     # z_obs = 0.001 # redshift of observation
#     t_obs = LCDM.age(z_obs).value  # age of the universe at this redshift

#     t_initial = 0.3  # Gyr, start the integration at this times
#     t_span = (t_initial, t_obs)  #  span of the integration
#     print("observed at z = {:.2f}".format(z_obs))

#     # what would be the masses of these halos at t_initial
#     mhalos_init = [initial_mhalo(mhalo_zobs, t_span) for mhalo_zobs in mhalos_zobs]
#     print(mhalos_init)
#     mhalos_init = np.array(mhalos_init) * u.Msun

#     galaxy_stellar_masses = []
#     galaxy_gas_masses = []
#     halo_masses = []
#     halo_sfe = []
#     for j, mhalo_t0 in enumerate(tqdm(mhalos_init)):

#         eta_m = 10 * (mhalos_zobs[j] / (1e10 * u.solMass)) ** -0.7
#         # eta_m =  20 * (mhalos_zobs[i] / (1e10 * u.solMass)) ** -0.7
#         eta_e = 0.15 * (mhalos_zobs[j] / (1e12 * u.solMass)) ** -0.5

#         model = CGM_regulator(
#             mhalo_t0, t_span, tstep=0.01, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z
#         )
#         run = model.run_halo()
#         results = model.get_results()
#         derived = model.get_derived_quantities()
#         # halo sfe
#         f_star = derived["f_star"]

#         # save final state per halo mass
#         galaxy_stellar_masses.append(results["m_star"][-1])
#         galaxy_gas_masses.append(results["m_gas"][-1])
#         halo_masses.append(results["m_halo"][-1])
#         halo_sfe.append(f_star[-1])

#     galaxy_gas_masses_zevol.append(galaxy_gas_masses)
#     galaxy_stellar_masses_zevol.append(galaxy_stellar_masses)
#     halo_masses_zevol.append(halo_masses)
#     f_stars.append(halo_sfe)

# galaxy_stellar_masses_zevol = np.array(galaxy_stellar_masses_zevol)
# galaxy_gas_masses_zevol = np.array(galaxy_gas_masses_zevol)
# halo_masses_zevol = np.array(halo_masses_zevol)
# f_stars = np.array(f_stars)
# # %%
# fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)

# ax.plot(
#     10**loghm,
#     10**logSMHM,
#     color="k",
#     ls="--",
#     label=r"Behroozi et al. 2019, $z \sim$ 0",
# )
# ax.fill_between(
#     10**loghm,
#     10 ** (logSMHM + SMHMerr[1, :]),
#     10 ** (logSMHM - SMHMerr[0, :]),
#     color="gray",
#     alpha=0.6,
# )
# ax.set(
#     xlabel=r"M$_{\rm halo}$ [M$_\odot$]",
#     ylabel=r"M$_\star$/M$_{\rm halo}$ [M$_\odot$]",
#     yscale="log",
#     xscale="log",
#     xlim=(1e10, 1e12),
#     # ylim=(1e-4, 0.05),
# )
# # lessthan_mw_mask = mhalos_z0.value < 1e12
# f_baryon = 0.16

# # make a colorbar spanning the z_array to color the lines with
# cmap = plt.get_cmap("viridis")
# norm = plt.Normalize(0, z_array[0])

# # # make a cbar axis on the top of the plot
# cbar_ax = ax.inset_axes([0.0, 1.04, 1, 0.03])
# cbar = plt.colorbar(
#     plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, orientation="horizontal"
# )
# cbar_ax.set_title("$z$")
# # reverse the cbar
# cbar.ax.invert_xaxis()
# # add the tick marks on the top instead of bottom of axis
# cbar.ax.xaxis.set_ticks_position("top")

# for i, z in enumerate(z_array):
#     smhm = galaxy_stellar_masses_zevol[i] / halo_masses_zevol[i]
#     ax.plot(
#         mhalos_zobs,
#         smhm,
#         color=cmap(norm(z)),
#         # label=r"$z = {:.1f}$".format(z),
#     )

#     ax.set(
#         xscale="log",
#         yscale="log",
#         xlabel=r"$M_{\rm halo}$",
#         ylabel=r"$M_{\star} / M_{\rm halo}$",
#     )

# ax.axhline(f_baryon, color="grey", ls=":", label="Cosmic baryon fraction")
# # title= r"$\eta_M$ = 20 (mhalo at z obs) / 1e10 Msun, $z_{{\rm obs}} = {:}$".format(z_obs)
# title = "varying $\eta_M$ and $\eta_E$"
# # ax.legend(title=title, loc="lower right")
# ax.legend(title=title, loc="lower left", bbox_to_anchor=(1.01, 0.05))

# plt.show()
# # %%
