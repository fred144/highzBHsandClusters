# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp
from astropy import cosmology
# import seaborn as sns
from regulator_lib.cooling_fn_generator import cooling_fn_generator
import astropy.constants as consts
import astropy.units as u

from tqdm import tqdm


from scipy.interpolate import RegularGridInterpolator

# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
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
        # "xtick.major.size": 6,
        # "ytick.major.size": 6,
        # "xtick.minor.size": 4,
        # "ytick.minor.size": 4,
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

def custom_mass_loading(mhalo, A=1, alpha=-1.4):
    """mass loading factor as a function of halo mass"""
    return A * (mhalo / (1e12 * u.solMass)) ** alpha


def custom_energy_loading(mhalo_z0, A=0.15, alpha=-0.5):
    """energy loading factor as a function of halo mass"""
    return A * (mhalo_z0 / (1e12 * u.solMass)) ** alpha


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
    eq. 13 from Carr 2023

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


def depletion_time_McGaugh(z, mstar):
    """depletion time fit from McGaugh observations

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
    if mdot <= 0:
        print("____", mhalo)

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
    """Halo virial temp

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


def density0(mCGM, r1, Rvir, alpha=1.4):
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
        (4 * np.pi * r1 ** (3) * ((Rvir / r1) ** (3 - alpha) - 1))
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
        eta_m=0.5,
        eta_e=0.1,
        eta_z=0.3,
        verbose=False,
    ):
        self.mhalo_z0 = mhalo_z0
        self.verbose = verbose
        # self.mhalo_init = mhalo_init

        self.time_interval = time_interval
        self.eta_m = eta_m
        self.eta_e = eta_e
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
        self.dep_time_norm = 0.4  # depletion time prefactor

        # sets how E_out is at preventing inflow (smaller - more effective)
        self.cgm_ejecti_specific_energy_ratio = 1

        # ratio of specific energy of ejected gas to CGM gas
        self.cgm_infall_prevention_const = 1

        # self.cooling_fn = cooling_fn_generator(
        #     "./tables/Lambda_tab_redshifts.npz"
        # )  # resides outside the class

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

    def mass_evolution(self, t, mass, ode_mode=True):
        """Ode to solve the mass evolution of the galaxy

        Args:
        t (_type_): _description_
        mass (_type_): _description_

        Returns:
        _type_: _description_
        """

        #  t: time (units: Gyr)
        #  mass [0-4] (units: solar mass): 5 vector of mass of each component/term
        #  energy [5-9] (units: erg): 5 vector of energy of each component/term

        # get current redshift
        z = cosmology.z_at_value(LCDM.age, t * u.Gyr)

        m_gas = mass[0] * u.solMass
        m_star = mass[1] * u.solMass
        m_cgm = mass[2] * u.solMass  # Total CGM mass
        m_metals = mass[3] * u.solMass
        m_halo = mass[4] * u.solMass  # total halo mass

        e_ism_wind = mass[5] * u.erg  # Energy gained from energy-loaded galactic winds
        # Energy loss from gas precipitation onto the Galaxy
        e_cgm_cool = mass[6] * u.erg
        # Energy loss from mass ejected from the CGM into IGM
        e_cgm_out = mass[7] * u.erg
        e_cgm_in = mass[8] * u.erg  # Energy gained from mass accretion from the IGM
        e_cgm = mass[9] * u.erg  # Total CGM energy

        t_depletion = depletion_time(z, m_star, self.exp, self.dep_time_norm)
        # can also use   # t_dep = depletion_time_McGaugh(z,mstar.value) * u.Gyr
        # halo properties
        halo_rvir = virial_radius(z, m_halo).to(u.kpc)
        halo_vir_temp = virial_T(m_halo, halo_rvir).to(u.K)

        # print("halo_vir_temp ",halo_vir_temp)
        t_dynamical = t_ff(halo_rvir, halo_rvir, m_halo).to(u.Gyr)
        rho_crit = LCDM.critical_density(z)
        cgm_metallicity = m_metals / m_cgm
        cgm_metallicity_sol = cgm_metallicity / self.Z_sol
        # cooling function value at this timestep

        # cooling_lambda = self.cooling_fn(
        #     (-4, np.log10(halo_vir_temp.value), cgm_metallicity_sol, 0)
        # ) * (u.erg * u.cm**3 * u.s**-1)
        # print("halo_vir_temp", halo_vir_temp)
        # print("cgm_metallicity_sol", cgm_metallicity_sol)
        cooling_lambda = self.cooling_fn.cooling_function(
            halo_vir_temp.value, cgm_metallicity_sol
        ) * (u.erg * u.cm**3 * u.s**-1)
        # print("***", halo_vir_temp.value, cgm_metallicity_sol, cooling_lambda)
        # print("cooling_lambda", cooling_lambda)

        if cooling_lambda < 0:
            print("cooling_lambda", cooling_lambda)
            cooling_lambda = 0 * (u.erg * u.cm**3 * u.s**-1)
            # raise ValueError("Negative cooling_lambda")

        # compute density normalization for power-law density model from CGM mass
        r1 = 0.1 * halo_rvir  # our definition of inner radius of CGM
        ## Another possible definition of inner radius of CGM
        # r1 = 10.0*u.kpc

        # get the central density in units of kpc^-3 ?
        rho0 = density0(mCGM=m_cgm, r1=r1, Rvir=halo_rvir, alpha=self.alpha)

        # estimate energy ejection loss timescale and limitit to dynamical time
        c_sound = np.sqrt(e_cgm / m_cgm)  # approximate sound speed
        t_ejection = halo_rvir / c_sound  # ejection time of hot gas
        # if ode_mode == False:
        #     print(t_ejection)
        #     print(t_dynamical)
        t_ejection = min(max(t_ejection, 0.1 * t_dynamical), t_dynamical)

        # outflow rate assuming that gas flows out at the sound speed
        # eq. 18 from Carr+ 2023
        # Energy ejection loss rate = (E_CGM - E_vir)/t_ej
        dot_e_cgm_out = (
            max(e_cgm - self.kb * halo_vir_temp * m_cgm / self.mu, 0.0 * u.erg)
            / t_ejection
        ).to(u.erg / u.Gyr)

        # radiative losses in the cgm, integrated, Eq. 3 from Carr 2023
        dot_e_cgm_cool = energy_loss(
            Lamb=cooling_lambda,
            Rvir=halo_rvir,
            r1=r1,
            rho0=rho0,
            mu=self.mu,
            alpha=self.alpha,
        ).to(u.erg / u.Gyr)

        # speceific energy of eject gas
        cgm_specific_e = self.cgm_ejecti_specific_energy_ratio * max(
            e_cgm / m_cgm, self.kb * halo_vir_temp / self.mu
        )

        # (effective) cooling time of CGM
        tcool_eff = (cgm_specific_e / dot_e_cgm_cool) * m_cgm

        if tcool_eff < 0:
            print(
                "Negative cooling time",
                cooling_lambda,
                cgm_specific_e,
                dot_e_cgm_cool,
                m_cgm,
            )
            raise ValueError("Negative cooling time")

        #   tcomp = 1.2e7*u.yr * ((1+20)/(1+z))**4
        #   tcool_eff = min(tcool_eff, tcomp)  # include Compton cooling
        tcool_eff = tcool_eff + 1 * t_dynamical  # include dynamical time?
        dot_e_cgm_cool = (e_cgm / tcool_eff).to(u.erg / u.Gyr)

        ##################### moving on to mass evolution
        # CGM_eject_dot = max(mCGM - mCGM_precip, 0 * u.solMass) / (t_dyn)
        # CGM_eject_dot = (energy_gain(eta_e, mstar_dot) * (mu/(kb*Tvir))).to(u.solMass/u.Gyr)

        # CGM mass loss due to cooling, eq. 7 from Carr 2023
        # print("tcool_eff", tcool_eff)
        dot_m_cgm_cool = (m_cgm / tcool_eff).to(u.solMass / u.Gyr)

        # eq. 12, m_gas is kinda the ism

        if t_depletion < 0:
            raise ValueError("Negative depletion time")

        dot_m_sfr = m_gas / t_depletion

        # eq 11 ish
        dot_m_gas = (dot_m_cgm_cool - dot_m_sfr * (1 + self.eta_m)).to(
            u.solMass / u.Gyr
        )
        dot_m_sfr = dot_m_sfr.to(u.solMass / u.Gyr)

        dot_m_halo = halo_infall_fakhouri(z, m_halo)

        dot_m_cgm_in = self.fb * dot_m_halo  # eq. 6
        # if (ode_mode == False) & (t > 0.5) & (t < 0.55):
        #     print("dot_mhalo: {:.2e}".format(dot_m_halo))
        #     print("m_halo:", m_halo)
        #     print("dot_m_cgm_in: {:.2e}".format(dot_m_cgm_in))
        #     print("halo_vir_temp: {:.2e}".format(halo_vir_temp))
        # CGM eject loss term, eq 10
        dot_m_cgm_out = ((1 / cgm_specific_e) * dot_e_cgm_out).to(u.solMass / u.Gyr)

        ################## calculate the energies that depend on the mass changes

        # ratio of dot ECGM_in / ECGM_out
        dot_energy_from_infall = (self.kb * halo_vir_temp / self.mu).to(
            u.erg / u.solMass
        ) * dot_m_cgm_in

        ## debugging
        # print([self.kb, halo_vir_temp, self.mu, dot_m_cgm_in])
        # print("dot_energy_from_infall",dot_energy_from_infall.to(u.erg * u.Gyr**-1))
        # print("dot_m_cgm_in",dot_m_cgm_in)
        # print("dot_e_cgm", dot_e_cgm_out)

        e_ejection_to_infall_ratio = self.cgm_infall_prevention_const / (
            dot_e_cgm_out / dot_energy_from_infall
        )

        ## debugging
        # if np.isnan(e_ejection_to_infall_ratio) or np.isinf(e_ejection_to_infall_ratio):
        #     e_ejection_to_infall_ratio = 0.1
        # print("e_ejection_to_infall_ratio", e_ejection_to_infall_ratio)
        # equation 19 CGM infall prevention factor
        f_prevent = min(max(e_ejection_to_infall_ratio, 0.1), 1.0)  # 0.1 < f < 1
        dot_m_cgm_in *= f_prevent
        # if (ode_mode == False) &(t > 0.5) & (t < 0.55):
        #     print("dot_m_cgm_in  {:.2e}".format(dot_m_cgm_in))
        #     print("f_prevent", f_prevent)

        #     print("-dot_e_cgm_out", dot_e_cgm_out)
        #     print("-dot_energy_from_infall", dot_energy_from_infall)
        #     print("-e_ejection_to_infall_ratio", e_ejection_to_infall_ratio)
        #     print("--", max(e_ejection_to_infall_ratio, 0.1))

        # energy input from SF
        dot_e_ism_wind = energy_gain(self.eta_e, dot_m_sfr)

        # energy due to accretion, eq 16
        dot_e_cgm_in = (self.kb * halo_vir_temp / self.mu * dot_m_cgm_in).to(
            u.erg * u.Gyr**-1
        )

        # CGM feedback gain term, eq 9
        dot_m_ism_wind = dot_m_sfr * self.eta_m

        ##################  total Mass, Energy, and Metallicity derivatives
        dot_m_cgm = dot_m_cgm_in + dot_m_ism_wind - dot_m_cgm_cool - dot_m_cgm_out
        dot_m_metal = (
            self.metal_yield * self.eta_z * dot_m_sfr
            + self.Z_IGM * self.Z_sol * dot_m_cgm_in
            - cgm_metallicity * (dot_m_cgm_cool + dot_m_cgm_out)
        )
        dot_e_cgm = (
            dot_e_ism_wind + 1 * dot_e_cgm_in - dot_e_cgm_out - 1 * dot_e_cgm_cool
        )

        # limiter

        ## smoothly go to 0
        ## half step point
        if self.verbose:
            print(" ")
            print("**time", t, "z", z)

            if dot_m_cgm_cool < dot_m_sfr:
                print("dot_m_cgm_cool < dot_m_sfr", dot_m_cgm_cool, dot_m_sfr)

            print("> m_gas, m_star, m_cgm, m_metals, m_halo")
            print(
                ">>  {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    m_gas, m_star, m_cgm, m_metals, m_halo
                )
            )
            print("> e_ism_wind, e_cgm_cool, e_cgm_out, e_cgm_in, e_cgm")
            print(
                ">>  {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    e_ism_wind, e_cgm_cool, e_cgm_out, e_cgm_in, e_cgm
                )
            )

            print("pre limited dot_m_cgm {:.2e}".format(dot_m_cgm))

        ##TODO: set mdot to 0 smoothly, instead of hard limit
        if (m_cgm.value < 1e3) & (dot_m_cgm < 0):
            dot_m_cgm *= max((m_cgm.value - 5e3) / 5e3, 0)
            print("dot_m_cgm < 0", dot_m_cgm)

        halo_sfe = m_star / (m_halo * self.fb)

        if self.verbose:
            # do some checks
            print("mass of the CGM: {:.2e}".format(m_cgm))
            print("post-limited dot_m_cgm: {:.2e}".format(dot_m_cgm))
            print("dot m CGM in: {:.2e}".format(dot_m_cgm_in))
            print("dot m ism wind: {:.2e}".format(dot_m_ism_wind))
            print("dot m cgm cool: {:.2e}".format(dot_m_cgm_cool))
            print("dot m cgm out: {:.2e}".format(dot_m_cgm_out))

            # print all timescales
            print("t_depletion: {:.2e}".format(t_depletion))
            print("t_dynamical: {:.2e}".format(t_dynamical))
            print("tcool_eff: {:.2e}".format(tcool_eff.to(u.Gyr)))
            print("t_ejection: {:.2e}".format(t_ejection))
            print("t_ff: {:.2e}".format(t_ff(halo_rvir, halo_rvir, m_halo).to(u.Gyr)))

            # print all the dot_ value

            print("> dot_m_gas, dot_m_sfr, dot_m_cgm, dot_m_metal, dot_m_halo")
            print(
                ">> {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    dot_m_gas, dot_m_sfr, dot_m_cgm, dot_m_metal, dot_m_halo
                )
            )

            print(
                "> dot_e_ism_wind, dot_e_cgm_cool, dot_e_cgm_out, dot_e_cgm_in, dot_e_cgm"
            )
            print(
                ">>  {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}".format(
                    dot_e_ism_wind,
                    dot_e_cgm_cool,
                    dot_e_cgm_out,
                    dot_e_cgm_in,
                    dot_e_cgm,
                )
            )

        # if dot_m_sfr is negative, stop
        # if m_gas < 0:
        #     raise ValueError("Negative ISM mass")

        if ode_mode:
            derivs = np.array(
                [
                    dot_m_gas.value,
                    dot_m_sfr.value,
                    dot_m_cgm.value,
                    dot_m_metal.value,
                    dot_m_halo.value,
                    dot_e_ism_wind.value,
                    dot_e_cgm_cool.value,
                    dot_e_cgm_out.value,
                    dot_e_cgm_in.value,
                    dot_e_cgm.value,
                ]
            )
            return derivs
        else:
            return np.array(
                [
                    halo_vir_temp.value,
                    dot_e_cgm_out.value,
                    dot_e_cgm_cool.value,
                    dot_e_cgm_in.value,
                    dot_e_ism_wind.value,
                    dot_m_cgm_out.value,
                    dot_m_cgm_cool.value,
                    dot_m_cgm_in.value,
                    dot_m_ism_wind.value,
                    f_prevent,
                    halo_sfe,
                    t_depletion.value,
                    t_dynamical.value,
                    tcool_eff.value,
                    t_ejection.value,
                    cooling_lambda.value,
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

        mass_cgm_0 = 1e3
        e_cgm_0 = (mass_cgm_0 * u.solMass * self.fb * self.kb * tvir / self.mu).to(
            u.erg
        )
        initial_cgm_metal_zsun = 0.001
        # get the corresponding metal mass
        mass_cgm_metals_0 = mass_cgm_0 * initial_cgm_metal_zsun * self.Z_sol
        # initial conditions masses and energ
        masses_initial = np.array(
            [
                1e3,
                1e3,
                mass_cgm_0,
                mass_cgm_metals_0,
                mhalo_t0,
                1e3,
                1e3,
                1e3,
                1e3,
                e_cgm_0.value,
            ]
        )
        print(
            "> initial values ISM gas mass = {:.2e} Msol\tStellar mass = {:.2e} Msol\tCGM mass = {:.2e} Msol\tMetal mass = {:.2e} Msol\tCGM metallicity = {:.4f} (Zsun)\tHalo mass = {:.2e} Msol\tISM wind energy = {:.2e} erg\tCGM cooling energy = {:.2e} erg\tCGM out energy = {:.2e} erg\tCGM in energy = {:.2e} erg\tCGM energy = {:.2e} erg".format(
                masses_initial[0],
                masses_initial[1],
                masses_initial[2],
                masses_initial[3],
                (masses_initial[3] / masses_initial[2]) / self.Z_sol,
                masses_initial[4],
                masses_initial[5],
                masses_initial[6],
                masses_initial[7],
                masses_initial[8],
                masses_initial[9],
            )
        )

        if self.tstep != 1:
            t = self.evaluation_time_array
            solution = solve_ivp(
                self.mass_evolution,
                self.time_interval,
                masses_initial,
                # method="RK45",
                rtol=1e-3,
                t_eval=t,
            )
        else:
            solution = solve_ivp(
                self.mass_evolution,
                self.time_interval,
                masses_initial,
                # method="RK45",
                rtol=1e-3,
                # atol=1e-6,
            )

        adaptive_tsteps = solution.t
        adaptive_z = cosmology.z_at_value(LCDM.age, adaptive_tsteps * u.Gyr)
        mgas_t = solution.y[0]
        mstar_t = solution.y[1]
        mcgm_t = solution.y[2]
        mmetals_t = solution.y[3]
        mhalo_t = solution.y[4]

        egy_t = solution.y[5]  # energy gained from energy-loaded galactic winds
        egy_radloss_t = solution.y[6]  # energy lost due to cooling
        egy_eject_t = solution.y[7]  # energy ejected from the CGM
        egy_accrete_t = solution.y[8]  # energy accreted from the IGM
        egy_cgm_t = solution.y[9]  # total cgm energy

        metal_cgm_mass = mmetals_t / mcgm_t  # CGM metallicity ratio
        metal_cgm_mass_sol = metal_cgm_mass / self.Z_sol

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
        print("_________________________________________")

        # fill the ode_results with the results
        self.ode_results["z"] = adaptive_z
        self.ode_results["t"] = adaptive_tsteps
        self.ode_results["m_gas"] = mgas_t
        self.ode_results["m_star"] = mstar_t
        self.ode_results["m_cgm"] = mcgm_t
        self.ode_results["m_metals"] = mmetals_t
        self.ode_results["m_halo"] = mhalo_t
        self.ode_results["egy_ism_wind"] = egy_t
        self.ode_results["egy_radloss"] = egy_radloss_t
        self.ode_results["egy_eject"] = egy_eject_t
        self.ode_results["egy_accrete"] = egy_accrete_t
        self.ode_results["egy_cgm"] = egy_cgm_t
        self.ode_results["metal_cgm_mass"] = metal_cgm_mass
        self.ode_results["metal_cgm_mass_sol"] = metal_cgm_mass_sol

        # print(
        #     "solution.t",
        #     len(adaptive_tsteps),
        #     "deriv_t",
        #     np.array(self.sim_time_out).shape,
        #     "self.evaluation_time_array",
        #     self.evaluation_time_array.shape,
        # )

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
                self.ode_results["m_gas"],
                self.ode_results["m_star"],
                self.ode_results["m_cgm"],
                self.ode_results["m_metals"],
                self.ode_results["m_halo"],
                self.ode_results["egy_ism_wind"],
                self.ode_results["egy_radloss"],
                self.ode_results["egy_eject"],
                self.ode_results["egy_accrete"],
                self.ode_results["egy_cgm"],
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
            "tvir": derived_quantities[:, 0],
            "dot_e_cgm_out": derived_quantities[:, 1],
            "dot_e_cgm_cool": derived_quantities[:, 2],
            "dot_e_cgm_in": derived_quantities[:, 3],
            "dot_e_ism_wind": derived_quantities[:, 4],
            "dot_m_cgm_out": derived_quantities[:, 5],
            "dot_m_cgm_cool": derived_quantities[:, 6],
            "dot_m_cgm_in": derived_quantities[:, 7],
            "dot_m_ism_wind": derived_quantities[:, 8],
            "f_prevent": derived_quantities[:, 9],
            "f_star": derived_quantities[:, 10],
            "t_dep": derived_quantities[:, 11],
            "t_dyn": derived_quantities[:, 12],
            "tcool_eff": derived_quantities[:, 13],
            "t_ejection": derived_quantities[:, 14],
            "cooling_lambda": derived_quantities[:, 15],
        }





def plot_halo_diagnostics(results, derived_quant, eta_m=0.5, eta_e=0.1, eta_z=0.3):

    t = derived_quant["sim_time"]
    tvir = derived_quant["tvir"]
    dot_e_cgm_out = derived_quant["dot_e_cgm_out"]
    dot_e_cgm_cool = derived_quant["dot_e_cgm_cool"]
    dot_e_cgm_in = derived_quant["dot_e_cgm_in"]
    dot_e_ism_wind = derived_quant["dot_e_ism_wind"]
    dot_m_cgm_out = derived_quant["dot_m_cgm_out"]
    dot_m_cgm_cool = derived_quant["dot_m_cgm_cool"]
    dot_m_cgm_in = derived_quant["dot_m_cgm_in"]
    dot_m_ism_wind = derived_quant["dot_m_ism_wind"]
    f_prevent = derived_quant["f_prevent"]
    f_star = derived_quant["f_star"]
    t_depletion = derived_quant["t_dep"]
    t_dynamical = derived_quant["t_dyn"]
    tcool_eff = derived_quant["tcool_eff"]
    t_ejection = derived_quant["t_ejection"]
    cooling_lambda = derived_quant["cooling_lambda"]

    adaptive_z = results["z"]
    t_adaptive = results["t"]
    mgas_t = results["m_gas"]
    mgas_t = results["m_gas"]
    mstar_t = results["m_star"]
    mcgm_t = results["m_cgm"]
    mmetals_t = results["m_metals"]
    mhalo_t = results["m_halo"]
    egy_t = results["egy_ism_wind"]
    egy_radloss_t = results["egy_radloss"]
    egy_eject_t = results["egy_eject"]
    egy_accrete_t = results["egy_accrete"]
    egy_cgm_t = results["egy_cgm"]
    metal_cgm_mass = results["metal_cgm_mass"]
    metal_cgm_mass_sol = results["metal_cgm_mass_sol"]

    colors = plt.get_cmap('Dark2')(np.linspace(0, 1, 8))

    fig, ax = plt.subplots(3, 3, figsize=(12, 9), dpi=300, sharex="row")
    ax = ax.flatten()

    plt.subplots_adjust(hspace=0.15, wspace=0.2)

    #   ax[0,0].plot(sol.t, gradients[0,:])
    # add minor tick marks
    ax[0].minorticks_on()
    ax[3].minorticks_on()

    ax[0].plot(t, dot_e_ism_wind, label="gain from SF winds", color="b")
    ax[0].plot(t, dot_e_cgm_cool, label="CGM cooling loss", color="r")
    ax[0].plot(t, dot_e_cgm_in, label=r"cosmic infall", color="g")
    ax[0].plot(t, dot_e_cgm_out, label=r"eject from energy-loading", color="orange")

    run_text = r"$ M_\odot, M_{{\rm halo}} (z = {:.2f})= {:.0e} ~ M_\odot, M_{{\rm halo}}(z = {:.2f}) = {:.0e}~ M_\odot$, $\eta_m = {:.1f}, \eta_e = {:.1f}, \eta_z = {:.1f}$".format(
        adaptive_z[0].value,
        mhalo_t[0],
        adaptive_z[-1].value,
        mhalo_t[-1],
        eta_m,
        eta_e,
        eta_z,
    )

    run_text = (
        r"ran from $z = {:.2f} - {:.2f}$, time = {:.2f} - {:.2f} Gyr"
        "\n"
        r"$\eta_m = {:.1f}$"
        " \n"
        r"$ \eta_e = {:.1f}$"
        " \n  "
        r"$\eta_z = {:.1f}$"
        "\n"
        r"and $M_{{\rm halo}}(z = {:.1f}) = {:.0e} ~ M_\odot$, $M_{{\rm halo}}(z = {:.1f}) = {:.0e} ~ M_\odot$, $M_{{\rm halo}}(z = {:.1f}) = {:.0e} ~ M_\odot$".format(
            adaptive_z[0].value,
            adaptive_z[-1].value,
            t[0],
            t[-1],
            eta_m,
            eta_e,
            eta_z,
            adaptive_z[0].value,
            mhalo_t[0],
            adaptive_z[-1].value,
            mhalo_t[-1],
            0,
            mhalo_at_z0(mhalo_t[-1] * u.Msun, adaptive_z[-1]),
        )
    )
    # put text on the corner left
    ax[0].text(
        0.05,
        1.2,
        run_text,
        transform=ax[0].transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    ax[0].set(
        yscale="log",
        # ylim=(1e52, dot_e_cgm_out[-1] * 10),
        ylabel=r"$\dot{E} ~ [{\rm erg \ Gyr^{-1}}]$",
    )
    ax[0].legend(
        ncols=1,
        # bbox_to_anchor=(0.5, 1.45),
        loc="lower right",
        frameon=False,
        # title=run_text,
        fontsize=7,
    )

    ax[1].plot(t, dot_m_ism_wind, label="from SF", color="b")
    ax[1].plot(t, dot_m_cgm_cool, label="cooling gas", color="r")
    ax[1].plot(t, dot_m_cgm_in, label=r"cosmic infall", color="g")
    ax[1].plot(t, dot_m_cgm_out, label=r"eject from energy-loading", color="orange")
    ax[1].legend(
        ncols=2,
        # bbox_to_anchor=(0.5, 1.45),
        loc="lower right",
        frameon=False,
        # title=run_text,
        fontsize=7,
    )
    ax[1].set(
        ylim=(1e2, dot_m_cgm_in[-1] * 10),
        xlim=(t[0] * 0.9, t[-1]),
        yscale="log",
        ylabel=r" $ \dot{M} ~ [{\rm M_{\odot} \ Gyr^{-1}}] $",
    )

    ax[2].plot(t_adaptive, mstar_t, label=r"$M_{\star}$", color=colors[0])
    ax[2].plot(t_adaptive, mgas_t, label=r"$M_{\rm gas}$", color=colors[1])
    ax[2].plot(t_adaptive, mcgm_t, label=r"$M_{\rm CGM}$", color=colors[3])
    ax[2].plot(t_adaptive, mhalo_t, label=r"$M_{\rm halo}$", color="k")
    ax[2].legend(ncols=2, frameon=False)
    ax[2].set_ylabel(r"M$_{\odot}$")

    # plot the energy terms
    ax[3].plot(t, egy_t, label=r"$E_{\rm ISM, winds}$", color=colors[0])
    ax[3].plot(t, egy_radloss_t, label=r"$E_{\rm CGM, cooling}$", color=colors[3])
    ax[3].plot(t, egy_accrete_t, label=r"$E_{\rm CGM, accrete}$", color=colors[1])
    ax[3].plot(t, egy_eject_t, label=r"$E_{\rm eject}$", color=colors[2])

    ax[3].plot(t, egy_cgm_t, label=r"$E_{\rm CGM}$", color="k")
    ax[3].set(yscale="log", ylabel=r"$E ~ [{\rm erg}]$", ylim=(1e50, egy_t[-1] * 10))
    ax[3].legend(ncols=2, frameon=False, loc="upper left", fontsize=7)
    ax[4].plot(t, f_star)
    ax[4].set_yscale("log")
    ax[4].set_ylabel(r"halo scale $f_\star$")
    ax[4].set_ylim(1e-3, 0.5)

    ax[5].plot(t, f_prevent)
    ax[5].set_ylim(0, 1.25)
    ax[5].set_ylabel(r"$f_{\rm prevent}$")

    mu = 0.6 * consts.m_p
    kb = consts.k_B
    Teff = ((egy_cgm_t * u.erg) / (mcgm_t * u.solMass) * (mu / kb)).to(u.K)
    # print(Teff)
    ax[6].plot(t_adaptive, Teff, label="Teff")
    ax[6].plot(t, tvir, label="Tvir")
    ax[6].set_ylabel("T [K]")
    ax[6].set_yscale("log")

    ax[6].set_ylim(300, 1e8)
    ax[6].legend(ncols=2, frameon=False)

    # plot the metals
    ax[7].plot(t, metal_cgm_mass_sol, label="CGM metallicity")
    ax[7].set_ylabel(r"CGM metallicity $[Z_{\odot}]$")
    ax[7].set_yscale("log")
    ax[7].set_xlabel("t [Gyr]")

    # plot the cooling function
    ax[8].plot(t, np.log10(cooling_lambda))
    ax[8].set_ylabel(r"log $\Lambda$ [erg cm$^3$ s$^{-1}$]")

    ax[0].set_xlim(t[0] * 0.9, t[-1])
    ax[2].set_xlim(t[0] * 0.9, t[-1])
    ax[2].set_ylim(1e3, mhalo_t[-1] * 1.2)
    ax[2].set_yscale("log")

    # make a twin redshift axis for the top row, using z
    for i in range(3):

        ax2 = ax[i].twiny()
        # make sure the ranges are the same

        ax2.plot(t, dot_e_cgm_out, color="k", alpha=0)
        ax2.set_xlim(t[0] * 0.9, t[-1])
        z_ticks = [100, 80, 60, 50, 40, 30, 15, 13, 11, 10, 9, 8, 7, 6]
        t_ticks = LCDM.age(z_ticks).value  # Convert redshift to corresponding time

        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels(["{:.0f}".format(z) for z in z_ticks])
        ax2.set_xlim(t[0] * 0.9, t[-1])

        ax2.set_xlabel(r"$z$")
        # ax2.minorticks_on()

    plt.show()


# # run a test model
# mhalo_z0 = 5e8 * u.Msun
# t_span = (0.1, 1)  # gyrs
# eta_m = 0.1
# eta_e = 0.1
# eta_z = 0.2
# mhalo_t0 = 5e7 * u.Msun
# model = CGM_regulator(
#     mhalo_z0, t_span, tstep=0.001, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z
# )
# run = model.run_halo()
# results = model.get_results()
# derived = model.get_derived_quantities()
# plot_halo_diagnostics(results, derived, eta_m=eta_m, eta_e=eta_e, eta_z=eta_z)

#%%

def density0(mCGM, r0, Rvir, alpha=1.4):
    denominator = ((4 * np.pi * r0**3) / (3 - alpha)) * (  (Rvir / r0)**(3-alpha) - 1)
    return mCGM / denominator

def power_profile(r, r1, rho0, alpha=1.4):

    return rho0 * (r / r1) ** (-alpha)


def exponential_disk(r, r_trunc_disk, rho_disk_central):
    """simple exponential disk profile"""
    return rho_disk_central * np.exp(-r / r_trunc_disk)


def disk_density0(mISM, r_trunc_disk, r_vir):
    """
    the total mass of the ism is given analytically by
    (r_trunc_disk - r_trunc_disk* exp(-Rvir/ r_trunc_disk )) rho_0

    here we solve for rho_0

    """
    return mISM / (r_trunc_disk * (1 - np.exp(-r_vir / (r_trunc_disk))))

def nfw_profile(r, rho_bulge, r1):
    return (rho_bulge / ( (r / r1)* ( 1 + (r / r1)) ** 2) )

def isothermal(r, rho0, r1):
    """isothermal profile"""
    return (rho0 / (1 + (r / r1)**2))

Rvir = 1
mISM = 1e8  # Msun
mCGM = 1e9  # Msun
r_inner = 0.1  # inner radius of the CGM in Rvir


cgm_alpha = 1.4
rho0 = density0(mCGM, r_inner, Rvir, cgm_alpha)  # central density of the CGM

# radial coordinates
r_full = np.linspace(0, Rvir, 1000)  # units of Rvir
r = np.linspace(0.1, Rvir, 100)  # units of Rvir
r_disk_evaluate = np.linspace(0, 0.1)  # disk scale radius in Rvir
r_trunc_disk = r_inner
rh0_disk = disk_density0(mISM, r_trunc_disk, Rvir)
disk_profile = exponential_disk(r_disk_evaluate, r_trunc_disk, rh0_disk)

# at the radius of the disk, we want the CGM density to match the disk density
matching_disk_density = power_profile(r_inner, r_inner, rho0, cgm_alpha)

# at the truncation radius of the disk
# we want the disk density to match the CGM density
scaling_factor = matching_disk_density / exponential_disk(
    r_inner, r_trunc_disk, rh0_disk
)


# we scale the disk density by this factor and leave the CGM density as is
rh0_disk = rho0 *np.e  # this is now the "true" central density of the disk

# calculate core in terms of the disk truncation
rho_bulge = rho0 * 4 * (Ob0/Omegam0)

rho_iso = rho0 * 2  # this is the central density of the isothermal profile

# Profiles
disk_profile = exponential_disk(r_disk_evaluate, r_trunc_disk, rh0_disk)
cgm_profile = power_profile(r, r_inner, rho0, 1.4)
bulge_profile = nfw_profile(r_full , rho_bulge, 1)    # bulge profile 
isothermal_profile = isothermal(r_full, rho_iso, r_inner)  # isothermal profile
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300)
ax.plot(
    r,
    cgm_profile,
    label=r"power law CGM $\rho_{{\rm CGM}} = \rho_0 (r/r_0)^{{{:.1f}}}$".format(cgm_alpha),
    lw=3
)
ax.plot(
    r_disk_evaluate,
    disk_profile,
    label=r"exponential disk $\rho_{\rm disk} = e \rho_0 \exp(-r/r_0)$",
    lw=3
)
ax.plot(
    r_full,
    isothermal_profile,
    "--r",
    label=r"isothermal $\rho_{\rm iso} = \rho_0  [1 + (r/r_0)^2]^{-1}$",
    lw=2
    )

ax.plot(r_full , bulge_profile * (Ob0/Omegam0), label=r"NFW $\times f_b$", lw=3, color="grey", alpha=0.5)




ax.set(xlabel=r"$r/R_{\rm vir}$", ylabel=r"$\rho(r)$ ", xscale="log", yscale="log", xlim=(0.01, 1), ylim=(2e7, 8e10))
ax.legend(loc="lower left", fontsize=8, frameon=False)


plt.savefig(
    "./figures/profile_explorations.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)  
plt.show()
# %%

# %% vary the eta_m


def vary_mass_loading(z_obs, eta_m, eta_e=0.1, eta_z=0.3):
    mhalos = np.geomspace(1e10, 1e12, 10) * u.Msun
    t_init = 0.1  # Gyr
    t_final = LCDM.age(z_obs).value  # Gyr
    t_span = (t_init, t_final)  # span of the integration
    mhalos_at_z_obs = []
    x = []
    y = []

    for mass_loading_vals in eta_m:
        mhalo_obs = []
        mstar_obs = []
        for midx, mhalo in enumerate(mhalos):
            mhalo_z0 = mhalo_at_z0(mhalo, z_obs)
            print("observing halo with mass {:.2e} at z = {:.2f}".format(mhalo, z_obs))
            print("mass of halo at z=0 is {:.2e}".format(mhalo_z0))
            gridmodel = CGM_regulator(
                mhalo_z0 * u.Msun,
                t_span,
                tstep=0.001,
                eta_m=mass_loading_vals,
                eta_e=eta_e,
                eta_z=eta_z,
            )
            run = gridmodel.run_halo()
            results = gridmodel.get_results()
            derived = gridmodel.get_derived_quantities()
            m_star = results["m_star"][-1]
            m_halo = results["m_halo"][-1]
            mhalo_obs.append(m_halo)
            mstar_obs.append(m_star)

        mhalo_obs = np.array(mhalo_obs)
        mstar_obs = np.array(mstar_obs)
        x.append(mhalo_obs)
        y.append(mstar_obs)

    return x, y


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
# # %% now vary eta_m and eta_e together, we want to observe these halos at a given redshift
# ##########################################################################################
# mhalos_zobs = np.geomspace(1e10, 1e12, 10) * u.Msun
# mass_loading_grid = np.geomspace(0.1, 10, 5)
# energy_loading_grid = np.geomspace(0.1, 1, 5)

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
