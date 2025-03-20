# %%
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
from regulator_lib.cooling_fn_generator import cooling_fn_generator
import astropy.constants as consts
import astropy.units as u
import seaborn as sns

LCDM = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)  # Astropy built-in cosmology
import astropy.units as u

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
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


# %% import cooling function curves

cooling_fn = cooling_fn_generator("./tables/Lambda_tab_redshifts.npz")


# Parameters for eta_m function
A = 1
beta = 0.7


def mass_loading(A, mhalo, alpha):
    """mass loading factor as a function of halo mass"""
    return A * (mhalo / (1e12 * u.solMass)) ** (-alpha)


# %% integrands and relevant functions
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
    v_circ = np.sqrt(G * mhalo / Rvir) * 1.3
    return r / v_circ


def depletion_time(z, mstar, exp, C):
    """depletion time (power-law),
    eq. 13 from Carr 2023

    Args:
        z (_type_): redshift
        mstar (_type_): mass of galaxy in solar mass
        exp (_type_): _description_
        C (_type_): _description_

    Returns:
        _type_: _description_
    """
    tH = (1 / LCDM.H(z=z)).to(u.Gyr)
    return C * tH * (mstar / (4e10 * u.solMass)) ** (-exp)


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


def halo_infall(z, mhalo):
    """halo mass inflows, from # d M_{halo} / dt (Dekel et al 2009)
    also, Carr 2023 Eq 4

    Args:
        z (_type_): _description_
        mhalo (_type_): _description_

    Returns:
        _type_: \dot{M}_{halo}
    """
    return (
        0.47
        * mhalo
        * (mhalo / (1e12 * u.solMass)) ** (0.15)
        * ((1 + z) / 3) ** (2.25)
        * u.Gyr ** (-1)
    ).to(u.solMass / u.Gyr)


def virial_T(mhalo, Rvir):
    """Halo virial temp

    Args:
        mhalo (_type_): halo mass
        Rvir (_type_): halo virial radius

    Returns:
        _type_: virial temperture
    """
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


# def density0(mCGM, alpha, r1, Rvir):  # density factor
#     return (mCGM * (3 - alpha)) / (
#         (4 * np.pi * r1 ** (3) * ((Rvir / r1) ** (3 - alpha) - 1))
#     )


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


def energy_loss(Lamb, Rvir, r1, rho0):
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
    return (
        (4 * np.pi * ((rho0 / mu) ** 2) * (r1**3))
        * (Lamb)
        * (((Rvir / r1) ** (3 - 2 * alpha) - 1) / (3 - 2 * alpha))
    )


def halo_mass_evol(t, mass):
    """
    # Halo mass evolution called by initial_mhalo to estimate initial z=6 halo mass
    """
    mhalo = mass * u.solMass
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    mhalo_dot = halo_infall(z, mhalo)
    return -mhalo_dot


def initial_mhalo(mhalo_z0, time_interval):
    # Finds initial z=6 halo mass for a given (z=0) halo mass
    mass_initial = np.array([mhalo_z0.value])
    sol_0 = solve_ivp(halo_mass_evol, time_interval, mass_initial)
    return sol_0.y[0][-1]  # z=6 halo mass


# def loss_integrand(x, alpha): # Integrand for calculating CGM mass loss
#    return x**(-2*alpha) * x**2

# def electron_density(Lamb, T, t_ff): # equation (1) from Voit et al 2015
#    return (3*kb*T) / (10 * Lamb * t_ff)

# def CGM_massloss_integrand(x, r1, alpha, rho0, Rvir, mhalo, Tvir, Lamb):
#    rho = rho0 * ((x*u.kpc)/r1)**(-alpha)
#    tcool = 2*((1.5 * mu * kb * Tvir) / (rho * Lamb)).to(u.Gyr)
#    tff = t_ff((x*u.kpc), Rvir, mhalo).to(u.Gyr)
#    return (x**(2-alpha)) / (tcool.value + 0*tff.value)

# def E_IN_integrand(x, Tvir, Rvir, mhalo, Lamb): # Integrand for calculating energy loss from the CGM
##    ne = electron_density(Lamb, Tvir, t_ff((x*u.kpc), Rvir, mhalo)).to(u.kpc**-3).value
#    tff = t_ff((x*u.kpc), Rvir, mhalo).to(u.Gyr)
#    ne = ((3*kb*Tvir) / (10 * Lamb * tff)).to(u.kpc**-3).value
#    return ne**2 * x**2

# def CGM_profile(x,Tvir, Rvir, mhalo, Lamb): # integrand for estimating halo mass from CGM density profile
#    ne_r = electron_density(Lamb, Tvir, t_ff((x*u.kpc), Rvir, mhalo)).to(u.kpc**-3).value
#    return ne_r*x**2


# def energy_loss(Lamb, Tvir, Rvir, mhalo): # radiative energy loss from the CGM
#    return (Lamb* 4*np.pi * (integrate.quad(E_IN_integrand, 10, Rvir.value, epsrel=1e-3,
#            args=(Tvir, Rvir, mhalo, Lamb))[0] * u.kpc**-3)).to(u.erg * u.Gyr**-1)
# %% now the main functions
def mass_evolution(t, mass):
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

    mgas = mass[0] * u.solMass
    mstar = mass[1] * u.solMass
    mCGM = mass[2] * u.solMass  # Total CGM mass
    mZM = mass[3] * u.solMass
    mhalo = mass[4] * u.solMass

    E_g = mass[5] * (u.erg)  # Energy gained from energy-loaded galactic winds
    E_l = mass[6] * (u.erg)  # Energy loss from gas precipitation onto the Galaxy
    E_ej = mass[7] * (u.erg)  # Energy loss from mass enjected from the CGM
    E_acc = mass[8] * (u.erg)  # Energy gained from mass accretion from the IGM
    E_CGM = mass[9] * (u.erg)  # Total CGM energy

    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift

    t_dep = depletion_time(z, mstar, exp, C)  # estimate of depletion time
    # t_dep = depletion_time_McGaugh(z,mstar.value) * u.Gyr
    t_dep_list.append(t_dep.value)

    Rvir = virial_radius(z, mhalo).to(u.kpc)  # virial radius
    Tvir = virial_T(mhalo, Rvir).to(u.K)  # Virial Temperature
    rho_crit = LCDM.critical_density(z)  # critical density
    t_dyn = t_ff(Rvir, Rvir, mhalo).to(u.Gyr)  # dynamical time
    # t_cool = 10 * t_ff(Rvir,Rvir, mhalo).to(u.Gyr) # cooling time from precipitation
    Z_m = mZM / mCGM  # CGM metallicity
    Z = Z_m / Z_sol  # CGM metallicity with respect to solar metallicity
    Lamb = cooling_fn((-4, np.log10(Tvir.value), Z, 0)) * (
        u.erg * u.cm**3 * u.s**-1
    )  # cooling function value

    print("interpolated lambda", Lamb.value)

    # Compute density normalization for power-law density model from CGM mass
    r1 = 0.1 * Rvir  # our definition of inner radius of CGM
    # r1 = 10.0*u.kpc # Another possible definition of inner radius of CGM

    rho0 = density0(mCGM=mCGM, alpha=alpha, r1=r1, Rvir=Rvir)  # central density of CGM

    # Energy ejection loss timescale
    cs = np.sqrt(E_CGM / mCGM)  # specific energy of CGM (approximate sound speed)
    t_ej = Rvir / cs  # ejection time of hot gas
    t_ej = min(max(t_ej, 0.1 * t_dyn), t_dyn)  # clip to range 0.1-1 t_dyn

    # Energy ejection loss rate = (E_CGM - E_vir)/t_ej
    Eej_dot = (max(E_CGM - kb * Tvir * mCGM / mu, 0.0 * u.erg) / t_ej).to(u.erg / u.Gyr)

    # Energy radiative loss
    # integrated radiative loss rate
    El_dot = energy_loss(Lamb, Rvir, r1, rho0).to(u.erg / u.Gyr)

    # specific energy of ejected gas
    CGM_specific_energy = CGM_eject_specific_energy_ratio * max(
        E_CGM / mCGM, kb * Tvir / mu
    )

    # (effective) cooling time of CGM
    tcool_eff = (CGM_specific_energy / El_dot) * mCGM
    #   tcomp = 1.2e7*u.yr * ((1+20)/(1+z))**4
    #   tcool_eff = min(tcool_eff, tcomp)  # include Compton cooling

    # include dynamical time?
    tcool_eff = tcool_eff + 1 * t_dyn
    El_dot = (E_CGM / tcool_eff).to(u.erg / u.Gyr)

    # CGM mass loss due to cooling (using effective cooling time)
    CGM_cool_dot = (mCGM / tcool_eff).to(u.solMass / u.Gyr)

    # Mass Terms
    # CGM_eject_dot = max(mCGM - mCGM_precip, 0 * u.solMass) / (t_dyn)
    # CGM_eject_dot = (energy_gain(eta_e, mstar_dot) * (mu/(kb*Tvir))).to(u.solMass/u.Gyr)
    mstar_dot = (mgas / t_dep).to(u.solMass / u.Gyr)
    mgas_dot = (-(mgas / t_dep) * (1 + eta_m) + CGM_cool_dot).to(u.solMass / u.Gyr)
    mhalo_dot = halo_infall(z, mhalo)

    # CGM eject loss term
    CGM_eject_dot = ((1 / CGM_specific_energy) * Eej_dot).to(u.solMass / u.Gyr)

    # CGM infall terms
    CGM_infall_dot = fb * mhalo_dot
    print("CGM_infall_dot",CGM_infall_dot)
    # CGM infall prevention factor: ratio of ejection energy / infall energy
    Eej_infall_ratio = CGM_infall_prevention_constant / (
        Eej_dot / ((kb * Tvir / mu) * CGM_infall_dot)
    ).to(u.dimensionless_unscaled)
    print("e",Eej_infall_ratio)
    fprevent = min(max(Eej_infall_ratio, 0.1), 1.0)  # clip to 0.1-1.0
    CGM_infall_dot *= fprevent

    # CGM feedback gain term
    CGM_gain_dot = mstar_dot * eta_m

    # Energy gain due to star formation
    Eg_dot = energy_gain(eta_e, mstar_dot)

    # Energy loss due to ejection
    Eacc_dot = ((kb * Tvir) / mu * CGM_infall_dot).to(u.erg * u.Gyr**-1)

    # totaol Mass, energy, and metallicity derivatives
    mCGM_dot = CGM_infall_dot + CGM_gain_dot - CGM_cool_dot - CGM_eject_dot
    #    print(mCGM_dot, CGM_infall_dot, CGM_gain_dot, CGM_cool_dot, CGM_eject_dot)
    mZM_dot = (
        eta_z * y * mstar_dot
        + Z_IGM * Z_sol * CGM_infall_dot
        - Z_m * (CGM_cool_dot + CGM_eject_dot)
    )
    ECGM_dot = Eg_dot + 1 * Eacc_dot - Eej_dot - 1 * El_dot

    if (mCGM.value < 3e5) & (mCGM_dot < 0):
        mCGM_dot *= max((mCGM.value - 5e3) / 5e3, 0)
    #        mCGM_dot = 0.0 * u.solMass/u.Gyr

    # Record some quantities for plotting later
    TT_list.append(t)  # Time [Gyr]
    Tvir_list.append(Tvir.value)

    Eej_dot_list.append(Eej_dot.value)
    El_dot_list.append(El_dot.value)
    Eacc_dot_list.append(Eacc_dot.value)
    Eg_dot_list.append(Eg_dot.value)

    CGM_eject_list.append(CGM_eject_dot.value)
    CGM_cool_list.append(CGM_cool_dot.value)
    CGM_infall_list.append(CGM_infall_dot.value)
    CGM_gain_list.append(CGM_gain_dot.value)

    fprevent_list.append(fprevent)
    fstar_list.append(mstar / (fb * mhalo))

    grad_array = np.array(
        [
            mgas_dot.value,
            mstar_dot.value,
            mCGM_dot.value,
            mZM_dot.value,
            mhalo_dot.value,
            Eg_dot.value,
            El_dot.value,
            Eej_dot.value,
            Eacc_dot.value,
            ECGM_dot.value,
        ]
    )

    #    plot_list.append( (t,grad_array,Tvir_list,fprevent))

    return grad_array


def bathtub(time_interval, mhalo_z0):
    """run the model for a given halo mass at z=0

    Args:
        time_interval (float): tuple, time interval for the model (units: Gyr)
        mhalo_z0 (float): stellar mass at z=0 (units: solar mass)

    Returns:
        (array): time, redshift, gas mass, stellar mass, CGM mass, halo mass, CGM metallicity, energy gain, energy loss, energy ejection, energy accretion, CGM energy
    """

    mhalo0 = initial_mhalo(mhalo_z0, time_interval)
    Rvir = virial_radius(0, mhalo0 * u.solMass).to(u.kpc)  # virial radius
    Tvir = virial_T(mhalo0 * u.solMass, Rvir).to(u.K)  # Virial Temperature
    mcgm0 = 1e3
    ecgm0 = (1e3 * u.solMass * fb * kb * Tvir / mu).to(u.erg)
    mass_initial = np.array(
        [1e3, 1e3, mcgm0, 1e1, mhalo0, 1e3, 1e3, 1e3, 1e3, ecgm0.value]
    )

    sol = solve_ivp(mass_evolution, time_interval, mass_initial, rtol=1e-3)
    t = sol.t
    mgas_t = sol.y[0]
    mstar_t = sol.y[1]
    mCGM_t = sol.y[2]
    mZM_t = sol.y[3]
    mhalo_t = sol.y[4]

    Eg_t = sol.y[5]
    El_t = sol.y[6]
    Eej_t = sol.y[7]
    Eacc_t = sol.y[8]
    ECGM_t = sol.y[9]

    Z_m = mZM_t / mCGM_t  # CGM metallicity ratio
    Z_cgm = Z_m / Z_sol  # CGM metallicity with respect to solar metallicity
    #   z = np.sqrt((28/t) - 1) - 1 # cosmological redshift rough approx
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    return (
        t,
        z,
        mgas_t,
        mstar_t,
        mCGM_t,
        mhalo_t,
        Z_cgm,
        Eg_t,
        El_t,
        Eej_t,
        Eacc_t,
        ECGM_t,
    )


# %%# Observed data from Behroozi et al 2019
smmr = np.loadtxt("./data/sm_averages_a1.002310.dat")
loghm = smmr[8:21, 0]  # halo mass
hm = 10**loghm * u.solMass

logSM = smmr[8:21, 7]  # stellar mass
sm = 10 ** logSM[8:21] * u.solMass

SMerr = np.vstack([smmr[8:21, 3], smmr[8:21, 2]])  # error array
logSMHM = smmr[8:21, 10]  # stellar mass / halo mass
SMHMerr = np.vstack([smmr[8:21, 12], smmr[8:21, 11]])  # error array
fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey="all", dpi=300)
ax.plot(10**loghm, 10**logSMHM, color="k", ls="--")
ax.fill_between(
    10**loghm,
    10 ** (logSMHM + SMHMerr[1, :]),
    10 ** (logSMHM - SMHMerr[0, :]),
    color="gray",
    alpha=0.6,
    label="Behroozi et al. 2019",
)
ax.set(
    xlabel=r"M$_{\rm halo}$ [M$_\odot$]",
    ylabel=r"M$_\star$/M$_{\rm halo}$ [M$_\odot$]",
    yscale="log",
    xscale="log",
)
ax.legend()
plt.show()

# %%


def explore_halo(mhalo_z0, eta_m_, eta_e_, time_interval):
    # time_interval = (0.56, 13.7)  # z=9 to z=0
    # if (time_interval is None):
    #     #z = 15 to z=0

    #    zinit = np.sqrt((28/time_interval[0]) - 1) - 1
    #    zfinal = np.sqrt((28/time_interval[1]) - 1) - 1
    zinit = cosmology.z_at_value(LCDM.age, time_interval[0] * u.Gyr)
    zfinal = cosmology.z_at_value(LCDM.age, time_interval[1] * u.Gyr)
    print("Running models from z = {:.2f} to z = {:.2f}".format(zinit, zfinal))
    print("Halo mass at z=0 = {:.2e}  Msol".format(mhalo_z0))

    beta = 0.9
    # eta_m = mass_loading(A, mhalo_z0, beta)

    colors = sns.color_palette()

    # Parameters
    global CGM_infall_prevention_constant, CGM_eject_specific_energy_ratio
    global eta_m, eta_e, eta_z
    eta_m = eta_m_
    eta_e = eta_e_
    CGM_infall_prevention_constant = (
        1.0  # sets how E_out is at preventing inflow (smaller - more effective)
    )
    CGM_eject_specific_energy_ratio = (
        1.0  # ratio of specific energy of ejected gas to CGM gas
    )

    global TT_list, Tvir_list, CB_list_1, t_dep_list, fprevent_list, fstar_list
    global CGM_gain_list, CGM_infall_list, CGM_cool_list, CGM_eject_list
    global Eej_dot_list, El_dot_list, Eg_dot_list, Eacc_dot_list
    global Z_gain_list, Z_infall_list, Z_cool_list, Z_eject_list
    global msdot_list, mgdot_list, mcdot_list, loss_list, CGMacc_list
    TT_list, Tvir_list, CB_list_1, t_dep_list, fprevent_list, fstar_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    CGM_gain_list, CGM_infall_list, CGM_cool_list, CGM_eject_list = ([], [], [], [])
    Eej_dot_list, El_dot_list, Eg_dot_list, Eacc_dot_list = ([], [], [], [])
    Z_gain_list, Z_infall_list, Z_cool_list, Z_eject_list = ([], [], [], [])
    msdot_list, mgdot_list, mcdot_list, loss_list, CGMacc_list = ([], [], [], [], [])

    mhalo0 = initial_mhalo(mhalo_z0, time_interval)
    Rvir = virial_radius(zfinal, mhalo0 * u.solMass).to(u.kpc)  # virial radius
    Tvir = virial_T(mhalo0 * u.solMass, Rvir).to(u.K)  # Virial Temperature
    mcgm0 = 1e3
    ecgm0 = (1e3 * u.solMass * fb * kb * Tvir / mu).to(u.erg)
    mass_initial = np.array(
        [1e3, 1e3, mcgm0, 1e1, mhalo0, 1e3, 1e3, 1e3, 1e3, ecgm0.value]
    )
    # time_interval = (0.04, 0.1)
    sol = solve_ivp(mass_evolution, time_interval, mass_initial, rtol=1e-3)

    gradients = np.zeros_like(sol.y)
    #    for i in range(len(sol.t)):
    #        print(i,len(sol.y[i]),sol.y[:,i])
    #      gradients[:,i] = mass_evolution(sol.t[i], sol.y[:,i])
    #     fprevent[i] = fprevent_func(sol.t[i,])

    t = sol.t
    mgas_t = sol.y[0]
    mstar_t = sol.y[1]
    mCGM_t = sol.y[2]
    mZM_t = sol.y[3]
    mhalo_t = sol.y[4]

    Eg_t = sol.y[5]
    El_t = sol.y[6]
    Eej_t = sol.y[7]
    Eacc_t = sol.y[8]
    ECGM_t = sol.y[9]

    Z_m = mZM_t / mCGM_t  # CGM metallicity ratio
    Z_cgm = Z_m / Z_sol  # CGM metallicity with respect to solar metallicity
    z = cosmology.z_at_value(LCDM.age, t * u.Gyr)
    #    z = np.sqrt((28/t) - 1) - 1 # cosmological redshift
    # print(sol.status)
    mcgm_dot = np.array(mcdot_list)

    print(r"M_halo = {:.4e}  Msol".format(mhalo_t[-1]))
    print(r"M_star = {:.4e}  Msol".format(mstar_t[-1]))
    print(r"f_star = {:.4e}".format(mstar_t[-1] / (fb * mhalo_t[-1])))
    print(r"M_gas = {:.4e} Msol".format(mgas_t[-1]))
    print(r"M_CGM = {:.4e}  Msol".format(mCGM_t[-1]))
    print("CGM Z = {:.2e}".format(Z_cgm[-1]))

    return_dict = {
        "TT_list": TT_list,
        "Tvir_list": Tvir_list,
        "CB_list_1": CB_list_1,
        "Eg_dot_list": Eg_dot_list,
        "El_dot_list": El_dot_list,
        "Eacc_dot_list": Eacc_dot_list,
        "Eej_dot_list": Eej_dot_list,
        "t": t,
        "mgas_t": mgas_t,
        "mstar_t": mstar_t,
        "mCGM_t": mCGM_t,
        "mZM_t": mZM_t,
        "mhalo_t": mhalo_t,
        "Eg_t": Eg_t,
        "El_t": El_t,
        "Eej_t": Eej_t,
        "Eacc_t": Eacc_t,
        "ECGM_t": ECGM_t,
        "Z_cgm": Z_cgm,
        "z": z,
        "CGM_cool_list": CGM_cool_list,
        "CGM_eject_list": CGM_eject_list,
        "CGM_gain_list": CGM_gain_list,
        "CGM_infall_list": CGM_infall_list,
        "fprevent_list": fprevent_list,
        "fstar_list": fstar_list,
    }
    print(return_dict.keys())
    return return_dict


# %%


def plot_halo_evolution(model):
    TT_list = model["TT_list"]
    Tvir_list = model["Tvir_list"]
    Eg_dot_list = model["Eg_dot_list"]
    El_dot_list = model["El_dot_list"]
    Eacc_dot_list = model["Eacc_dot_list"]
    Eej_dot_list = model["Eej_dot_list"]

    t = model["t"]
    CGM_cool_list = model["CGM_cool_list"]
    CGM_eject_list = model["CGM_eject_list"]
    CGM_gain_list = model["CGM_gain_list"]
    CGM_infall_list = model["CGM_infall_list"]
    fprevent_list = model["fprevent_list"]
    fstar_list = model["fstar_list"]
    mgas_t = model["mgas_t"]
    mstar_t = model["mstar_t"]
    mCGM_t = model["mCGM_t"]
    mZM_t = model["mZM_t"]
    mhalo_t = model["mhalo_t"]
    ECGM_t = model["ECGM_t"]
    Z_cgm = model["Z_cgm"]
    z = model["z"]
    colors = sns.color_palette()

    fig, ax = plt.subplots(2, 3, figsize=(11, 5), dpi=300, sharex="row")
    ax = ax.flatten()
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    #   ax[0,0].plot(sol.t, gradients[0,:])
    # add minor tick marks
    ax[0].minorticks_on()
    ax[3].minorticks_on()

    ax[0].plot(TT_list, Eg_dot_list, label="gain from SF", color="b")
    ax[0].plot(TT_list, El_dot_list, label="cooling loss", color="r")
    ax[0].plot(TT_list, Eacc_dot_list, label=r"cosmic infall", color="g")
    ax[0].plot(
        TT_list, Eej_dot_list, label=r"eject from energy-loading", color="orange"
    )
    ax[0].legend(ncols=4, bbox_to_anchor=(0.0, 1.35), loc="upper left", frameon=False)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e52, 1e62)
    ax[0].set_ylabel(r"log $\dot{E}$")
   

    ax[1].plot(TT_list, CGM_gain_list, label="mass gain from SF", color="b")
    ax[1].plot(TT_list, CGM_cool_list, label="cooling gas", color="r")
    ax[1].plot(TT_list, CGM_infall_list, label=r"cosmic infall", color="g")
    ax[1].plot(
        TT_list, CGM_eject_list, label=r"eject from energy-loading", color="orange"
    )
    # ax[1].legend()
    ax[1].set_ylim(1e7, 1e12)
    ax[1].set_xlim(TT_list[0], TT_list[-1])
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r" $\dot{M}_{\odot}$")
    

    ax[2].plot(TT_list, fprevent_list)
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel(r"$f_{\rm prevent}$")
    

    ax[3].plot(TT_list, fstar_list)
    ax[3].set_ylim(1e-4, 1)
    ax[3].set_yscale("log")
    ax[3].set_ylabel(r"$f_\star$")
    ax[3].set_xlabel("t [Gyr]")

    ax[4].plot(t, np.log10(mstar_t), label=r"$M_{\star}$", color=colors[0])
    ax[4].plot(t, np.log10(mgas_t), label=r"$M_{\rm gas}$", color=colors[1])
    ax[4].plot(t, np.log10(mCGM_t), label=r"$M_{\rm CGM}$", color=colors[3])
    ax[4].plot(t, np.log10(mhalo_t), label=r"$M_{\rm halo}$", color="k")
    ax[4].legend(ncols=2, frameon=False)
    ax[4].set_ylabel(r"log M$_{\odot}$")
    ax[4].set_xlabel("t [Gyr]")
    ax[4].set_ylim(2, 12.4)

    Teff = ((ECGM_t * u.erg) / (mCGM_t * u.solMass) * (mu / kb)).to(u.K)
    ax[5].plot(t, Teff, label="Teff")
    ax[5].plot(TT_list, Tvir_list, label="Tvir")
    ax[5].set_ylabel("T (K)")
    ax[5].set_yscale("log")
   
    ax[5].set_ylim(300, 1e8)
    ax[5].legend(ncols=2, frameon=False)

    # make a twin redshift axis for the top row, using z
    for i in range(3):
        
        ax2 = ax[i].twiny()
        # make sure the ranges are the same
        ax2.set_xlim(ax[0].get_xlim())
        ax2.plot(TT_list, Eg_dot_list, color="k", alpha=0)
        # now get the tick labels and replace them with redshifts
        current_ticks = ax2.get_xticks()
        print(current_ticks)
        current_ticklabels = ax2.get_xticklabels()
        # z_ticks = [
        #     cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in current_ticks
        # ]
        # ax2.set_xticks(current_ticks)
        # ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
        # ax2.set_xlabel(r"$z$")
        ax2.minorticks_on()

    plt.show()


# %% run a model
# %% model parameters
# 1D galaxy model that produces a self-consistent model of
# the gas, stellar, and CGM contents for galaxies.
# Takes bounds on time and initial values for masses as inputs.

eta_m = 0.5  # mass-loading factor
eta_e = 0.1  # energy-loading factor
eta_z = 0.3  # metallicity-loading factor
y = 0.02  # metallicity enrichment yield (mass in metals produced per solar mass)
fb = 0.16  # universal baryon fraction
Delc = 200  # overdensity
G = consts.G  # Gravitational Constant
kb = consts.k_B  # Boltzmann constant
mp = consts.m_p  # mass of a proton (Hydrogen mass)
Z_sol = 0.0134  # solar metallicty
Z_IGM = 0.01  # Assumed metallicity of IGM in solar units
mu = 0.6 * mp  # mean molecular weight
alpha = 1.4  # density power-law index
exp = 0.0  # T_depletion power-law
C = 0.4

# controls how effective outflowing energy is at preventing inflowing gas
CGM_infall_prevention_constant = 1.0
# ratio of specific energy of ejected gas to CGM gas
CGM_eject_specific_energy_ratio = 1.0
# time, virial temperature, cooling function, depletion time, inflow prevention factor, star formation efficiency
TT_list, Tvir_list, CB_list_1, t_dep_list, fprevent_list, fstar_list = (
    [],
    [],
    [],
    [],
    [],
    [],
)
#
CGM_gain_list, CGM_infall_list, CGM_cool_list, CGM_eject_list = ([], [], [], [])
Eej_dot_list, El_dot_list, Eg_dot_list, Eacc_dot_list = ([], [], [], [])
Z_gain_list, Z_infall_list, Z_cool_list, Z_eject_list = ([], [], [], [])
msdot_list, mgdot_list, mcdot_list, loss_list, CGMacc_list = ([], [], [], [], [])

time_interval = (0.1, 1.2)
mh1e9t_004_05 = explore_halo(1e9 * u.solMass, 0.3, 0.15, time_interval)
plot_halo_evolution(mh1e9t_004_05)


# %%
# time_interval = (0.1, 13.4)
# mh1e11t_01_10 = explore_halo(1e11 * u.solMass, 3, 0.3, time_interval)
# plot_halo_evolution(mh1e11t_01_10)
