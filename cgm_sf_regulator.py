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
import warnings

from tqdm import tqdm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import time

# defined some customization
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

# standard flat cosmology
H0 = 70
h = 0.7
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
f_baryon = Ob0 / Omegam0  # universal baryon fraction
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)


# %%
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


class WiersamaCooling:
    def __init__(self, table_path="./tables/Lambda_tab_redshifts.npz"):
        # Load cooling table
        self.cooling_table = np.load(table_path)

        # Build interpolator once
        points = (
            self.cooling_table["log_nHbins"],
            self.cooling_table["log_Tbins"],
            self.cooling_table["Zs"],
            self.cooling_table["redshifts"],
        )
        self.interp = RegularGridInterpolator(
            points,
            self.cooling_table["Lambda_tab"],
            bounds_error=False,
            fill_value=None,
        )

        # Store bounds for clipping
        self.bounds = {
            "log_T": (
                self.cooling_table["log_Tbins"].min(),
                self.cooling_table["log_Tbins"].max(),
            ),
            "log_nH": (
                self.cooling_table["log_nHbins"].min(),
                self.cooling_table["log_nHbins"].max(),
            ),
            "Z": (
                self.cooling_table["Zs"].min(),
                self.cooling_table["Zs"].max(),
            ),
            "z": (
                self.cooling_table["redshifts"].min(),
                self.cooling_table["redshifts"].max(),
            ),
        }
        print("Cooling table bounds:")
        for key, val in self.bounds.items():
            print(f"  {key}: {val}")

    def __call__(self, density, temperature, metallicity, redshift):
        """Evaluate cooling function at given state."""
        # round the metallicity and temperature to 0.0001
        input_metallicity = np.round(metallicity, 4)
        input_temperature = np.round(temperature, 0)

        input_log_T = np.log10(input_temperature)
        input_log_nH = np.log10(density)
        print("_______________")
        print(
            "**** log_nH={:.2e}, log_T={:.2e}, metallicity={:.2e}, redshift={:.2f}".format(
                input_log_nH, input_log_T, input_metallicity, redshift
            )
        )

        # Clip to table range
        log_T = np.clip(input_log_T, *self.bounds["log_T"])
        metallicity = np.clip(
            input_metallicity, self.bounds["Z"][0], self.bounds["Z"][1]
        )
        log_nH = np.clip(input_log_nH, *self.bounds["log_nH"])
        z = np.clip(redshift, *self.bounds["z"])

        # Check for clipping
        if input_log_T != log_T:
            print(f"clipping log_T: input={input_log_T:.3e}, clipped={log_T:.3e}")
        if input_metallicity != metallicity:

            print(
                f"clipping metallicity: input={input_metallicity:.3e}, clipped={metallicity:.3e}"
            )
        if log_nH != input_log_nH:
            print(f"clipping log_nH: input={input_log_nH:.3e}, clipped={log_nH:.3e}")
        if z != redshift:
            print(f"clipping redshift: input={redshift:.3f}, clipped={z:.3f}")

        # interpolate
        Lambda = self.interp((log_nH, log_T, metallicity, z))

        if np.isnan(Lambda):
            raise ValueError(
                f"Cooling function is NaN at "
                f"(log_nH={log_nH}, log_T={log_T}, Z={metallicity}, z={z})"
            )

        return Lambda


# %%test out the Weirsama cooling
# cooling_fn = WiersamaCooling()
# density = 1e-4
# temperature = 1e4
# metallicity = 1
# redshift = 10
# cooling_lambda = cooling_fn(density, temperature, metallicity, redshift)
# print("lambda: ergs/s cm^3", cooling_lambda)


# %%


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


def halo_mass_growth_fakhouri(t, mass):
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


def halo_mass_growth_dekel(t, mass):
    """
    # Halo mass evolution called by initial_mhalo to
    # estimate initial halo mass at arbitrary redshift
    """
    mhalo = mass * u.solMass
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift
    if t > 13.47:  # 13.4 is the age of the universe in Gyr
        t = 13.466983947061877
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    mhalo_dot = halo_infall_dekel(z, mhalo)

    return mhalo_dot


def mhalo_at_z0_dekel(mhalo_at_z, z_obs):
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
    sol_0 = solve_ivp(halo_mass_growth_dekel, time_interval, mass_initial)
    return sol_0.y[0][-1]  # return the last value of the solution


def mhalo_at_z0_fakhouri(mhalo_at_z, z_obs):
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
    sol_0 = solve_ivp(halo_mass_growth_dekel, time_interval, mass_initial)
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
    # if rad_loss < 0:
    #     print("Negative radiative", Lamb, rho0, mu, r1, Rvir, alpha)
    return rad_loss


def halo_mass_evol_fakhouri(t, mass):
    """
    # Halo mass evolution called by
    # initial_mhalo to estimate initial halo mass at arbitrary redshift
    """
    mhalo = mass * u.solMass
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    mhalo_dot = halo_infall_fakhouri(z, mhalo)
    return -mhalo_dot


def halo_mass_evol_dekel(t, mass):
    """
    # Halo mass evolution called by
    # initial_mhalo to estimate initial halo mass at arbitrary redshift
    """
    mhalo = mass * u.solMass
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    mhalo_dot = halo_infall_dekel(z, mhalo)
    return -mhalo_dot


def initial_mhalo_fakhouri(mhalo_z0, time_interval):
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
    sol_0 = solve_ivp(halo_mass_evol_fakhouri, time_interval, mass_initial)
    return sol_0.y[0][-1]  # return the last value of the solution


def initial_mhalo_dekel(mhalo_z0, time_interval):
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
    sol_0 = solve_ivp(halo_mass_evol_dekel, time_interval, mass_initial)
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


class CGMRegulator:
    """CGM regulator with strictly SF


    Args:
            mhalo_z0 (_type_): _description_
        time_interval (_type_): _description_
        tstep (int, optional): _description_. Defaults to 1.
        eta_z (float, optional): _description_. Defaults to 0.3.
        verbose (bool, optional): _description_. Defaults to True.
        dep_time_norm (float, optional): _description_. Defaults to 0.4.
        cooling_dynamic_time_norm (int, optional): _description_. Defaults to 1.
        disk_scale_length (float, optional): _description_. Defaults to 0.02.
        KS_n (float, optional): _description_. Defaults to 1.5.
        KS_kappa_s (float, optional): _description_. Defaults to 0.1.
        r_bulge (int, optional): _description_. Defaults to 1.
        dbug_norm_for_accretion_energy_rate (int, optional): _description_. Defaults to 1.
        updated_halo_infall (bool, optional): _description_. Defaults to True.
        updated_loadings (bool, optional): _description_. Defaults to True.
        updated_2phase_CGM (bool, optional): _description_. Defaults to True.
        updated_SF_law (bool, optional): _description_. Defaults to True.
        add_f_prevent (bool, optional): _description_. Defaults to True.
        alpha_m (float, optional): _description_. Defaults to 0.1.
        alpha_e (float, optional): _description_. Defaults to 0.1.

    Yields:
        _type_: _description_


    Example:
        mhalo_z0 = 1e10 * u.Msun
        t_span = (0.1, 1)  # gyrs
        model = CGMRegulator(
            mhalo_z0,
            t_span,
            eta_z=0.3,
            disk_scale_length=0.02,
        )
        run_2phase = model.run_halo()

        # get the mass and energy evolution
        results_2phase = model.get_results()
        print(results_2phase.keys())

        # get derive quantities such as derivs or cooling rates
        derived_2phase = model.get_derived_quantities()
        print(derived_2phase.keys())

    """

    def __init__(
        self,
        mhalo_z0,
        time_interval,
        tstep=1,
        eta_z=0.3,
        verbose=True,
        dep_time_norm=0.4,
        cooling_dynamic_time_norm=1,
        disk_scale_length=0.02,
        KS_n=1.5,
        KS_kappa_s=0.1,
        r_bulge=1,
        # ep_bh_radeff=0.1,
        # ep_bh_feedback_eff=0.02,
        # bondi_boost=100,
        # r_grav_physical=1,
        # m_bh_seed = 1e2,
        dbug_norm_for_accretion_energy_rate=1,
        dbug_norm_for_2_phase_CGM=0,
        updated_halo_infall=True,
        updated_loadings=True,
        updated_2phase_CGM=True,
        updated_SF_law=True,
        add_f_prevent_floor=0.1,
        add_f_prevent_constant=None,
        alpha_m=0.1,
        alpha_e=0.1,
        custom_cooling_params=None,
    ):

        # fmt: off
        self.mhalo_z0 = mhalo_z0 # the mass of the halo we want to run to z=0
        self.verbose = verbose # extra printout
        self.time_interval = time_interval # simulation time in Gyr
        self.tstep = tstep # if set to 1, it is automatically dictated by the solver
        self.updated_halo_infall = updated_halo_infall
        self.updated_loadings = updated_loadings
        self.updated_2phase_CGM = updated_2phase_CGM
        self.updated_SF_law = updated_SF_law
        
        self.evaluation_time_array = np.arange(
            self.time_interval[0], self.time_interval[1], self.tstep
        )

        self.eta_z = eta_z # metal loading
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
        self.grav_cooling_time_norm = 1  # cooling time prefactor for accretion
        self.cooling_time_norm = cooling_dynamic_time_norm # cooling time prefactor for the cooling of CGM

        self.t_eject_lim_norm = 0.1  # ejection time limit factor
        self.cgm_ejecti_specific_energy_ratio = 1    # sets how E_out is at preventing inflow (smaller - more effective)
        self.cgm_infall_prevention_const = 1    # ratio of specific energy of ejected gas to CGM gas

        self.disk_scale = disk_scale_length  # disk scale radius in units of Rvir
        self.r_cgm_scale = 0.1  # inner radius of CGM in units of Rvir

        # KS law star formation
        self.ks_n = KS_n
        self.ks_kappa_s = KS_kappa_s
        self.r_bulge = r_bulge  # kpc
        
        # feedback loadings
        self.eta_M = alpha_m
        self.eta_E = alpha_e

        # self.cooling_fn = cooling_fn_generator(
        #     "./tables/Lambda_tab_redshifts.npz"
        # )  # resides outside the class
        
        # # BH params
        # self.r_grav_physical = r_grav_physical # kpc, accretion radius
        # self.ep_bh_radeff = ep_bh_radeff # fraction turned into feedback
        # self.ep_bh_feedback_eff = ep_bh_feedback_eff # feedback efficiency
        # self.bondi_boost = bondi_boost
        # self.mbh_seed = m_bh_seed
        
        # cooling table
        self.cooling_fn = CoolingFunctionInterpolator("./tables/newcool_viraj.dat")
        # self.cooling_fn = WiersamaCooling("./tables/Lambda_tab_redshifts.npz")

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
        
        # dbug, temporary parms
        self.dbug_norm_for_accretion_energy_rate = dbug_norm_for_accretion_energy_rate
        self.add_f_prevent_floor = add_f_prevent_floor
        self.add_f_prevent_constant = add_f_prevent_constant
        self.dbug_norm_for_2_phase_CGM = dbug_norm_for_2_phase_CGM
        self.custom_cooling_params = custom_cooling_params
        # fmt: on

    def mass_evolution(self, t, mass, ode_mode=True):
        """Ode to solve the mass evolution of the galaxy

        #  t: time (units: Gyr)
        #  mass [0-4] (units: solar mass): 5 vector of mass of each component/term
        #  energy [5-9] (units: erg): 5 vector of energy of each component/term

        This variant does the bulk of computations unitless (numerical values in
        chosen consistent units) to reduce astropy Quantity overhead. Inputs are
        assumed to be numeric values in:
            - masses: Msun
            - energies: erg
            - lengths when produced here: kpc (unless explicitly converted)
            - times: Gyr
        """

        # --- constants and conversion factors (numeric)
        sec_per_Gyr = u.Gyr.to(u.s)
        Msun_g = consts.M_sun.to(u.g).value
        kpc_to_cm = u.kpc.to(u.cm)
        cm_per_km = 1e5
        kb_erg_per_K = consts.k_B.to(u.erg / u.K).value
        mu_g = (0.6 * consts.m_p).to(u.g).value

        # get current redshift
        z = cosmology.z_at_value(LCDM.age, t * u.Gyr)

        # Unwrap inputs into unitless numeric values (Msun, erg)
        m_ism = float(mass[0])  # Msun
        m_star = float(mass[1])  # Msun
        m_bulge = float(mass[2])  # Msun
        m_cgm_hot = float(mass[3])  # Msun
        m_cgm_cold = float(mass[4])  # Msun
        m_cgm = m_cgm_hot + m_cgm_cold  # Msun
        m_metals = float(mass[5])  # Msun (metal mass in Msun)
        m_halo = float(mass[6])  # Msun
        e_ism_wind = float(mass[7])  # erg
        e_cgm_cool = float(mass[8])  # erg
        e_cgm_out = float(mass[9])  # erg
        e_cgm_in = float(mass[10])  # erg
        e_cgm = float(mass[11])  # erg

        # m_bh = mass[12] * u.solMass        # Total black hole mass

        # ------ derive halo properties using the helper functions but convert once
        halo_rvir_kpc = virial_radius(z, m_halo * u.solMass).to(u.kpc).value
        r1_kpc = self.r_cgm_scale * halo_rvir_kpc
        halo_vcirc_kms = (
            circular_velocity(m_halo * u.solMass, halo_rvir_kpc * u.kpc)
            .to(u.km / u.s)
            .value
        )
        halo_vir_temp_K = (
            virial_T(m_halo * u.solMass, halo_rvir_kpc * u.kpc).to(u.K).value
        )

        # updated loadings (unitless scalars)
        if self.updated_loadings:
            self.eta_m = vcirc_mass_loading(
                halo_vcirc_kms * u.km / u.s, alpha_m=self.eta_M
            )
            self.eta_e = vcirc_energy_loading(
                halo_vcirc_kms * u.km / u.s, alpha_e=self.eta_E
            )
        else:
            self.eta_m = custom_mass_loading(m_halo * u.solMass, A=10, alpha=-0.7)
            self.eta_e = custom_energy_loading(m_halo * u.solMass, A=0.1, alpha=-0.4)

        # ensure scalar numeric
        try:
            eta_e_val = float(self.eta_e)
        except Exception:
            eta_e_val = np.asarray(self.eta_e).astype(float)
        if np.any(eta_e_val > 1):
            eta_e_val = np.where(eta_e_val > 1, 1.0, eta_e_val)
        self.eta_e = eta_e_val

        # compute CGM effective temperature (numeric K)
        # e_cgm [erg], m_cgm_hot [Msun] -> convert mass to g -> energy per gram (erg/g)
        mass_hot_g = max(m_cgm_hot, 1e-12) * Msun_g
        e_per_mass_erg_per_g = e_cgm / mass_hot_g if mass_hot_g > 0 else 0.0
        # cgm_temp = (e_cgm/m_cgm_hot) * (mu / kb)  -> numeric K:
        cgm_temp_K = e_per_mass_erg_per_g * (mu_g / kb_erg_per_K)

        # dynamical time estimate (use kpc and km/s -> seconds -> Gyr)
        # t_dynamical = r / v
        r_km = halo_rvir_kpc * (u.kpc.to(u.km))
        t_dynamical_sec = r_km / max(halo_vcirc_kms, 1e-12)
        t_dynamical_Gyr = t_dynamical_sec / sec_per_Gyr

        # CGM metallicity (in solar units)
        m_cgm_safe = max(m_cgm, 1e-12)
        cgm_metallicity = m_metals / m_cgm_safe  # metal mass fraction
        cgm_metallicity_sol = cgm_metallicity / self.Z_sol

        # ----- cooling lambda (unit: erg cm^3 s^-1) numeric
        if self.custom_cooling_params is not None:
            cooling_lambda_val = float(
                self.cooling_fn.custom_cooling_function(
                    cgm_temp_K,
                    cgm_metallicity_sol,
                    T_thresh_cool=self.custom_cooling_params[0],
                    T_slope=self.custom_cooling_params[1],
                )
            )
        else:
            cooling_lambda_val = float(
                self.cooling_fn.cooling_function(cgm_temp_K, cgm_metallicity_sol)
            )

        # ----- density normalization rho0 (we compute unitless numeric in Msun/kpc^3)
        rho0_msun_per_kpc3 = density0(
            mCGM=m_cgm_hot, r0=r1_kpc, Rvir=halo_rvir_kpc, alpha=self.alpha
        )
        # convert rho0 to g/cm^3 for cgs calculations
        rho0_g_cm3 = rho0_msun_per_kpc3 * Msun_g / (kpc_to_cm**3)

        # central number density (1/cm^3)
        nh_0 = rho0_g_cm3 / mu_g if mu_g > 0 else 0.0

        # ----- compute radiative energy loss integrated numerically in CGS
        # energy_loss (cgs): erg/s
        # r1 and Rvir convert to cm
        r1_cm = r1_kpc * kpc_to_cm
        Rvir_cm = halo_rvir_kpc * kpc_to_cm
        # cooling_lambda_val is erg cm^3 s^-1
        # rad_loss_cgs = 4*pi * (rho0/mu)^2 * r1^3 * Lambda * ((Rvir/r1)^(3-2alpha) - 1) / (3-2alpha)
        # where rho0 and mu in g/cm^3 and g respectively -> (rho0/mu) in 1/cm^3 consistent with Lambda cgs
        term_geom = ((Rvir_cm / r1_cm) ** (3 - 2 * self.alpha) - 1.0) / (
            3 - 2 * self.alpha
        )
        if np.isfinite(term_geom):
            rad_loss_erg_per_s = (
                4.0
                * np.pi
                * (rho0_g_cm3 / mu_g) ** 2
                * (r1_cm**3)
                * cooling_lambda_val
                * term_geom
            )
        else:
            rad_loss_erg_per_s = 0.0
        # convert to erg/Gyr for consistency with rest of code (time units are Gyr)
        dot_e_cgm_hot_loss_erg_per_Gyr = rad_loss_erg_per_s * sec_per_Gyr

        # ----- ejection timescale and limits
        # compute sound speed from e_per_mass (erg/g) -> cm/s
        c_sound_cm_s = np.sqrt(max(e_per_mass_erg_per_g, 0.0))
        c_sound_kms = c_sound_cm_s / cm_per_km
        # t_ejection = rvir / c_sound (in seconds -> Gyr)
        t_ejection_sec = (halo_rvir_kpc * kpc_to_cm) / max(c_sound_cm_s, 1e-12)
        t_ejection_Gyr = t_ejection_sec / sec_per_Gyr

        # apply limits: between self.t_eject_lim_norm * t_dynamical and t_dynamical
        t_ejection_Gyr = min(
            max(t_ejection_Gyr, self.t_eject_lim_norm * t_dynamical_Gyr),
            t_dynamical_Gyr,
        )

        # dot_e_cgm_out: energy ejected per Gyr (erg/Gyr)
        # compute energy excess relative to thermal content at virial temp:
        # thermal_energy_virial = kb * Tvir * (mass_hot_g) / mu_g (erg)
        thermal_energy_virial = kb_erg_per_K * halo_vir_temp_K * mass_hot_g / mu_g
        e_excess_erg = max(e_cgm - thermal_energy_virial, 0.0)
        dot_e_cgm_out_erg_per_Gyr = e_excess_erg / max(t_ejection_Gyr, 1e-30)

        # ----- cooling time and energy cooling rate (Gyr units)
        # cgm_specific_e: erg per g (use max between e_per_mass and kb*T/mu)
        kb_term_erg_per_g = kb_erg_per_K * cgm_temp_K / mu_g if mu_g > 0 else 0.0
        cgm_specific_e_erg_per_g = self.cgm_ejecti_specific_energy_ratio * max(
            e_per_mass_erg_per_g, kb_term_erg_per_g
        )
        # total energy associated with that specific energy for the hot mass (erg)
        total_energy_erg = cgm_specific_e_erg_per_g * mass_hot_g
        # tcool (Gyr) = total_energy / dot_e_cgm_hot_loss (erg/Gyr)
        if dot_e_cgm_hot_loss_erg_per_Gyr <= 0:
            tcool_Gyr = (
                (1.0 / LCDM.H(z=z)).to(u.Gyr).value
            )  # fallback to Hubble time numeric
        else:
            tcool_Gyr = total_energy_erg / dot_e_cgm_hot_loss_erg_per_Gyr

        # prevent non-physical or negative
        tcool_Gyr = max(tcool_Gyr, 1e-12)

        # dot_e_cgm_cooling (erg/Gyr)
        dot_e_cgm_cooling_erg_per_Gyr = e_cgm / tcool_Gyr

        # ----- mass exchange rates (Msun/Gyr)
        dot_m_cgm_hot_cooling = m_cgm_hot / tcool_Gyr
        dot_m_cgm_cold_falling = m_cgm_cold / max(t_dynamical_Gyr, 1e-30)

        # star formation (KS-like) -- do unitless arithmetic (Msun, kpc)
        r_disk_kpc = self.disk_scale * halo_rvir_kpc
        sigma0 = m_ism / (2.0 * np.pi * (r_disk_kpc**2))  # Msun / kpc^2
        Asfr = 1e-12 * self.ks_kappa_s * 1e9  # msun / Gyr / kpc^2 as in prior code
        if self.verbose:
            print(
                f"dot_m_star inputs: Asfr={Asfr:.3e}, sigma0={sigma0:.3e}, ks_n={self.ks_n:.3e}, r_disk={r_disk_kpc:.3e}, m_ism={m_ism:.3e}"
            )
        dot_m_star = (
            Asfr * (sigma0**self.ks_n) * (2.0 * np.pi * r_disk_kpc**2) / (self.ks_n**2)
        )  # Msun / Gyr

        if self.updated_SF_law:
            dot_m_sfr = dot_m_star  # Msun / Gyr
        else:
            t_depletion = depletion_time(
                z, m_star * u.solMass, self.exp, self.dep_time_norm
            )
            dot_m_star_alt = (
                (m_ism * u.solMass / t_depletion).to(u.solMass / u.Gyr).value
            )
            dot_m_sfr = dot_m_star_alt

        # bulge SFR (numeric)
        dot_m_bulge = (2.0 * np.pi * Asfr * sigma0**self.ks_n) / (self.ks_n**2)
        dot_m_bulge *= r_disk_kpc**2 - np.exp(
            -self.ks_n * self.r_bulge / r_disk_kpc
        ) * (r_disk_kpc**2 + self.ks_n * self.r_bulge * r_disk_kpc)
        dot_mstar_central = dot_m_bulge  # Msun / Gyr

        # ISM mass rate (two-phase mixing debug handling preserved)
        dbug_rate = (
            (1.0 - self.dbug_norm_for_2_phase_CGM) * dot_m_cgm_cold_falling
            + self.dbug_norm_for_2_phase_CGM * dot_m_cgm_hot_cooling
        )
        dot_m_ism = dbug_rate - dot_m_sfr * (1.0 + float(self.eta_m))

        # halo infall (Msun/Gyr)
        if self.updated_halo_infall:
            dot_m_halo = (
                halo_infall_fakhouri(z, m_halo * u.solMass).to(u.solMass / u.Gyr).value
            )
        else:
            dot_m_halo = (
                halo_infall_dekel(z, m_halo * u.solMass).to(u.solMass / u.Gyr).value
            )

        dot_m_cgm_in = self.fb * dot_m_halo  # Msun / Gyr

        # mass ejected from CGM due to energy outflow (Msun/Gyr)
        # dot_m_cgm_out = (1 / cgm_specific_e) * dot_e_cgm_out
        # cgm_specific_e is in erg/g, dot_e_cgm_out in erg/Gyr -> result g/Gyr convert to Msun/Gyr
        if cgm_specific_e_erg_per_g > 0:
            dot_m_cgm_out_g_per_Gyr = (
                dot_e_cgm_out_erg_per_Gyr / cgm_specific_e_erg_per_g
            )
            dot_m_cgm_out = dot_m_cgm_out_g_per_Gyr / Msun_g
        else:
            dot_m_cgm_out = 0.0

        # prevention due to ejection vs infall energy (numeric)
        dot_energy_from_infall_erg_per_Gyr = (kb_erg_per_K * halo_vir_temp_K / mu_g) * (
            dot_m_cgm_in * Msun_g
        )
        e_ejection_to_infall_ratio = self.cgm_infall_prevention_const * (
            dot_energy_from_infall_erg_per_Gyr / max(dot_e_cgm_out_erg_per_Gyr, 1e-30)
        )

        if self.add_f_prevent_floor:
            f_prevent = float(
                np.clip(e_ejection_to_infall_ratio, self.add_f_prevent_floor, 1.0)
            )
        else:
            f_prevent = 1.0

        if self.add_f_prevent_constant is not None:
            f_prevent = float(self.add_f_prevent_constant)

        dot_m_cgm_in *= f_prevent

        # BH routines preserved in comments (untouched)
        ##################### BH routines
        # c_s = np.sqrt(self.kb * cgm_temp / self.mu).to(u.km / u.s)
        # rho_0_hot = rho0 # can relax this assumption maybe, cuspy?
        # dot_m_bh_bondi = (4 * np.pi * self.G**2 * rho_0_hot * m_bh**2) / c_s**3
        # dot_m_bh_bondi = self.bondi_boost * dot_m_bh_bondi.to(u.solMass / u.Gyr)
        #
        # ...
        ##################### BH routines end

        # energy input from SF (erg/Gyr)
        dot_e_ism_wind_erg_per_Gyr = (
            energy_gain(self.eta_e, dot_m_sfr * u.solMass / u.Gyr)
            .to(u.erg / u.Gyr)
            .value
        )
        # energy due to accretion (erg/Gyr)
        dot_e_cgm_in_erg_per_Gyr = (kb_erg_per_K * halo_vir_temp_K / mu_g) * (
            dot_m_cgm_in * Msun_g
        )  # erg/Gyr

        # CGM feedback gain/loss terms
        dot_m_ism_wind = dot_m_sfr * float(self.eta_m)  # Msun/Gyr

        if self.verbose:
            print(f"dot_m_sfr {dot_m_sfr:.3e}")
            print(
                f"At z={z:.2f}, m_halo={m_halo:.3e}, dot_m_halo={dot_m_halo:.3e}, f_prevent={f_prevent:.3f}"
            )
            print(
                f"m_ism={m_ism:.3e}, m_star={m_star:.3e}, m_bulge={m_bulge:.3e}, m_cgm_hot={m_cgm_hot:.3e}, m_cgm_cold={m_cgm_cold:.3e}, m_metals={m_metals:.3e}"
            )
            print(
                f"e_ism_wind={e_ism_wind:.3e}, e_cgm_cool={e_cgm_cool:.3e}, e_cgm_out={e_cgm_out:.3e}, e_cgm_in={e_cgm_in:.3e}, e_cgm={e_cgm:.3e}"
            )
            print(
                f"dot_m_ism={dot_m_ism:.3e}, dot_m_sfr={dot_m_sfr:.3e}, dot_m_bulge={dot_m_bulge:.3e}"
            )
            print(
                f"dot_e_cgm_out={dot_e_cgm_out_erg_per_Gyr:.3e}, dot_e_cgm_hot_loss={dot_e_cgm_hot_loss_erg_per_Gyr:.3e}, dot_e_cgm_in={dot_e_cgm_in_erg_per_Gyr:.3e}, dot_e_cgm_cooling={dot_e_cgm_cooling_erg_per_Gyr:.3e}"
            )

        # --- main derivatives (unitless numeric arrays matching original units)
        dot_m_cgm_hot = (
            dot_m_cgm_in + dot_m_ism_wind - dot_m_cgm_hot_cooling - dot_m_cgm_out
        )
        dot_m_cgm_cold = dot_m_cgm_hot_cooling - dot_m_cgm_cold_falling

        dot_e_cgm = (
            dot_e_ism_wind_erg_per_Gyr
            + self.dbug_norm_for_accretion_energy_rate * dot_e_cgm_in_erg_per_Gyr
            - dot_e_cgm_out_erg_per_Gyr
            - 1.0 * dot_e_cgm_hot_loss_erg_per_Gyr
            # + dot_e_bh_thermfeedback
        )

        dot_m_metal = (
            self.metal_yield * self.eta_z * dot_m_sfr
            + self.Z_IGM * self.Z_sol * dot_m_cgm_in
            - cgm_metallicity * dot_m_cgm_hot_cooling
            - cgm_metallicity * dot_m_cgm_out
        )

        # safety guards for low CGM masses
        if (m_cgm_hot < 5e3) and (dot_m_cgm_hot < 0):
            dot_m_cgm_hot *= max((m_cgm_hot - 5e3) / 5e3, 0.0)
            warning_message = (
                f"dot_m_cgm_hot={dot_m_cgm_hot:.3e} (m_cgm_hot={m_cgm_hot:.3e})"
            )
            # warnings.warn(
            #     warning_message,
            # )
        if (m_cgm_cold < 5e3) and (dot_m_cgm_cold < 0):
            dot_m_cgm_cold *= max((m_cgm_cold - 5e3) / 5e3, 0.0)
            warning_message = (
                f"dot_m_cgm_cold={dot_m_cgm_cold:.3e} (m_cgm_cold={m_cgm_cold:.3e})"
            )
            # warnings.warn(
            #     warning_message,
            # )

        halo_sfe = m_star / (m_halo * self.fb) if (m_halo * self.fb) > 0 else 0.0

        if self.verbose:
            print(
                f"dot_m_cgm_hot_cooling={dot_m_cgm_hot_cooling:.3e} (m_cgm_hot={m_cgm_hot:.3e}, tcool_real={tcool_Gyr:.3e})"
            )
            print(f"  tcool_real calculation:")
            print(f"    cgm_specific_e={cgm_specific_e_erg_per_g:.3e}")
            print(f"    dot_e_cgm_hot_loss={dot_e_cgm_hot_loss_erg_per_Gyr:.3e}")

            print(
                f"dot_m_cgm_cold_falling={dot_m_cgm_cold_falling:.3e} (m_cgm_cold={m_cgm_cold:.3e}, t_dynamical={t_dynamical_Gyr:.3e})"
            )

            print(
                f"*** m_cgm_hot={m_cgm_hot:.3e}, m_cgm_cold={m_cgm_cold:.3e}, t_dynamical={t_dynamical_Gyr:.3e}"
            )
            print(f"dot_m_cgm_hot={dot_m_cgm_hot:.3e}")
            print("dot_m_cgm_hot is composed of:")
            print(f"  dot_m_cgm_in         {dot_m_cgm_in:.3e}")
            print(f"  dot_m_ism_wind       {dot_m_ism_wind:.3e}")
            print(f"  dot_m_cgm_hot_cooling {dot_m_cgm_hot_cooling:.3e}")
            print(f"    m_cgm_hot={m_cgm_hot:.3e}")
            print(f"    tcool_real={tcool_Gyr:.3e}")
            print(f"    tcool_real is calculated from:")
            print(f"        cgm_specific_e = {cgm_specific_e_erg_per_g:.3e}")
            print(
                f"        dot_e_cgm_hot_loss (the integrand) = {dot_e_cgm_hot_loss_erg_per_Gyr:.3e}"
            )
            print(f"            Lamb={cooling_lambda_val:.3e}")
            print(f"            Rvir={halo_rvir_kpc:.3e} kpc")
            print(f"            r1={r1_kpc:.3e} kpc")
            print(f"            rho0={rho0_msun_per_kpc3:.3e} Msun/kpc^3")
            print(f"            mu={mu_g:.3e} g")
            print(f"            alpha={self.alpha:.3e}")
            print(f"  dot_m_cgm_hot_cooling = m_cgm_hot / tcool_real")
            print(f"  dot_m_cgm_out        {dot_m_cgm_out:.3e}")

        # Return either ODE derivatives (unitless matching initial array units) or derived quantities
        if ode_mode:
            derivs = np.array(
                [
                    float(dot_m_ism),  # Msun / Gyr
                    float(dot_m_sfr),  # Msun / Gyr
                    float(dot_mstar_central),  # Msun / Gyr
                    float(dot_m_cgm_hot),  # Msun / Gyr
                    float(dot_m_cgm_cold),  # Msun / Gyr
                    float(dot_m_metal),  # Msun / Gyr
                    float(dot_m_halo),  # Msun / Gyr
                    float(dot_e_ism_wind_erg_per_Gyr),  # erg / Gyr
                    float(dot_e_cgm_cooling_erg_per_Gyr),  # erg / Gyr
                    float(dot_e_cgm_out_erg_per_Gyr),  # erg / Gyr
                    float(dot_e_cgm_in_erg_per_Gyr),  # erg / Gyr
                    float(dot_e_cgm),  # erg / Gyr
                    # dot_m_bh.value,
                    # dot_e_bh_thermfeedback.value,
                ]
            )
            return derivs
        else:
            # Derived quantities: all unitless numeric values
            return np.array(
                [
                    float(halo_vir_temp_K),
                    float(dot_e_cgm_out_erg_per_Gyr),
                    float(dot_e_cgm_cooling_erg_per_Gyr),
                    float(dot_e_cgm_in_erg_per_Gyr),
                    float(dot_e_ism_wind_erg_per_Gyr),
                    float(dot_m_cgm_out),
                    float(
                        dot_m_cgm_hot
                    ),  # Msun / Gyr: hot bucket that feeds the cold buckets
                    float(dot_m_cgm_cold),  # Msun / Gyr: cold bucket that feeds the ISM
                    float(dot_m_cgm_in),
                    float(dot_m_ism_wind),
                    float(f_prevent),
                    float(halo_sfe),
                    float(t_dynamical_Gyr),
                    float(tcool_Gyr),
                    float(cooling_lambda_val),
                    float(halo_rvir_kpc),
                    float(dot_m_sfr),
                    float(dot_mstar_central),
                    float(t_ejection_Gyr),
                    float(cgm_temp_K),
                    float(rho0_msun_per_kpc3),
                    float(dot_m_halo),
                    # dot_m_bh.value,
                    # dot_e_bh_thermfeedback.value,
                    # dot_m_bh_bondi.value,
                    # dot_m_bh_tla.value,
                    # m_gas_in_bh_disk,
                    # m_star_in_bh_disk,
                    # effective_bulge_mass,
                ]
            )

    def run_halo(self):
        print("self.time_interval", self.time_interval, "tstep", self.tstep)
        if self.updated_halo_infall:
            mhalo_t0 = initial_mhalo_fakhouri(self.mhalo_z0, self.time_interval)
        else:
            mhalo_t0 = initial_mhalo_dekel(self.mhalo_z0, self.time_interval)

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

        e_bh_feedback_0 = 0.0 * u.erg
        # initial conditions masses and energ
        initial_values = np.array(
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
                # self.mbh_seed.value,
                # e_bh_feedback_0.value,
            ]
        )
        print(
            "> initial values ISM  = {:.2e} Msol \t Stellar mass = {:.2e} Msol\tBulge stellar mass = {:.2e} Msol\tCGM hot mass = {:.2e} Msol\tCGM cold mass = {:.2e} Msol\tMetal mass = {:.2e} Msol\tCGM metallicity = {:.4f} (Zsun)\tHalo mass = {:.2e} Msol\tISM wind energy = {:.2e} erg\tCGM cooling energy = {:.2e} erg\tCGM out energy = {:.2e} erg\tCGM in energy = {:.2e} erg\tCGM energy = {:.2e} erg".format(
                initial_values[0],
                initial_values[1],
                initial_values[2],
                initial_values[3],
                initial_values[4],
                initial_values[5],
                (initial_values[5] / mass_cgm_0) / self.Z_sol,
                initial_values[6],
                initial_values[7],
                initial_values[8],
                initial_values[9],
                initial_values[10],
                initial_values[11],
            )
        )

        t0 = time.perf_counter()
        print(f"Starting ODE solver (perf_counter={t0:.6f})")

        if self.tstep != 1:  # custom timesteping
            t = self.evaluation_time_array
            solution = solve_ivp(
                self.mass_evolution,
                self.time_interval,
                initial_values,
                # method="RK45",
                # rtol=1e-5,
                t_eval=t,
            )
        else:  # automatic timesteping
            solution = solve_ivp(
                self.mass_evolution,
                self.time_interval,
                initial_values,
                # rtol=1e-5,
            )

        elapsed = time.perf_counter() - t0
        print(
            f"ODE solver finished in {elapsed:.3f} s (status={getattr(solution, 'status', None)}, nfev={getattr(solution, 'nfev', None)})"
        )

        adaptive_tsteps = solution.t
        adaptive_z = cosmology.z_at_value(LCDM.age, adaptive_tsteps * u.Gyr)

        #### solutions
        mgas_t = solution.y[0]
        mstar_t = solution.y[1]
        mass_bulge_t = solution.y[2]
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
        # mbh = solution.y[12]
        # e_bh = solution.y[13]

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
        # self.ode_results["m_bh"] = mbh
        # self.ode_results["egy_bh"] = e_bh

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
                # self.ode_results["m_bh"],
                # self.ode_results["egy_bh"],
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
            "halo_vir_temp": derived_quantities[:, 0],
            "dot_e_cgm_out": derived_quantities[:, 1],
            "dot_e_cgm_cooling": derived_quantities[:, 2],
            "dot_e_cgm_in": derived_quantities[:, 3],
            "dot_e_ism_wind": derived_quantities[:, 4],
            "dot_m_cgm_out": derived_quantities[:, 5],
            "dot_m_cgm_hot": derived_quantities[
                :, 6
            ],  # hot bucket that feeds the cold buckets
            "dot_m_cgm_cold": derived_quantities[
                :, 7
            ],  # cold bucket that feeds the ISM
            "dot_m_cgm_in": derived_quantities[:, 8],
            "dot_m_ism_wind": derived_quantities[:, 9],
            "f_prevent": derived_quantities[:, 10],
            "halo_sfe": derived_quantities[:, 11],
            "t_dynamical": derived_quantities[:, 12],
            "tcool_real": derived_quantities[:, 13],
            "cooling_lambda": derived_quantities[:, 14],
            "halo_rvir": derived_quantities[:, 15],
            "dot_m_sfr": derived_quantities[:, 16],
            "dot_mstar_central": derived_quantities[:, 17],
            "t_ejection": derived_quantities[:, 18],
            "cgm_temp": derived_quantities[:, 19],
            "rho0": derived_quantities[:, 20],
            "dot_m_halo": derived_quantities[:, 21],
            # "dot_m_bh": derived_quantities[:, 18],
            # "dot_e_bh_thermfeedback": derived_quantities[:, 19],
            # "dot_m_bh_bondi": derived_quantities[:, 20],
            # "dot_m_bh_tla": derived_quantities[:, 21],
            # "m_gas_in_bh_disk": derived_quantities[:, 22],
            # "m_star_in_bh_disk": derived_quantities[:, 23],
            # "effective_bulge_mass": derived_quantities[:, 24],
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
            sci_notation_tex(
                mhalo_at_z0_fakhouri(mhalo_t[-1] * u.Msun, adaptive_z[-1])
            ),
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
            sci_notation_tex(
                mhalo_at_z0_fakhouri(mhalo_t[-1] * u.Msun, adaptive_z[-1])
            ),
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
            sci_notation_tex(
                mhalo_at_z0_fakhouri(mhalo_t[-1] * u.Msun, adaptive_z[-1])
            ),
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
    c_bh = "k"  # cmapset_2(1)

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
            sci_notation_tex(
                mhalo_at_z0_fakhouri(mhalo_t[-1] * u.Msun, adaptive_z[-1])
            ),
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
    ax[2].plot(t, dot_m_bh, label=r"$\dot{M}_{\rm \bullet}$", color=c_bh, lw=2)
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
        bbox_to_anchor=(-0.2, -0.05),
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


# mhalo_z0 = 1e10 * u.Msun
# t_span = (0.1, 1)  # gyrs

# ##fmt: off
# model = CGMRegulator(
#     mhalo_z0,
#     t_span,
#     eta_z=0.3,
#     disk_scale_length=0.02,
# )
# run_2phase = model.run_halo()
# %%
