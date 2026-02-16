# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from astropy import cosmology
import scipy
import cmasher as cmr
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator

# import seaborn as sns
from regulator_lib.cooling_fn_generator import cooling_fn_generator
import astropy.constants as consts
import astropy.units as u
import warnings
from cgm_sf_regulator import CGMRegulator
from tqdm import tqdm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import time
from matplotlib.colors import SymLogNorm
from regulator_lib.cooling_fn_generator import CoolingFunctionInterpolator


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)


# %%
# standard flat cosmology
H0 = 70
h = 0.7
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
f_baryon = Ob0 / Omegam0  # universal baryon fraction
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)

cooling_fn = CoolingFunctionInterpolator("./tables/newcool_viraj.dat")


def format_sci_notation(num, decimals=2):
    """format a number as LaTeX scientific notation with specified decimal places.

    Args:
        num: Number to format
        decimals: Number of decimal places (default: 2)

    Returns:
        string formatted as LaTeX scientific notation
    """
    if num == 0:
        return "0"

    exponent = int(np.floor(np.log10(np.abs(num))))
    mantissa = num / (10**exponent)

    # Round mantissa and adjust exponent if it rounds to 10
    mantissa = round(mantissa, decimals)
    if mantissa >= 10:
        mantissa /= 10
        exponent += 1

    return f"{mantissa:.{decimals}f} \\times 10^{{{exponent}}}"


def mdot_halo(z, mhalo):
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


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def virial_radius(z, mhalo, Delc=200):
    """Halo virial radius, classical 200 topahat overdensity
    Args:
        z (_type_): _description_
        mhalo (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (mhalo / (LCDM.critical_density(z) * (4 / 3) * np.pi * Delc)) ** (1 / 3)


def vcirc_energy_loading(halo_vcirc, alpha_e=0.1):
    eta_e = alpha_e * (halo_vcirc.value / 200) ** (-3 / 2)

    # if eta e > 1 set to 1, halo_vcirc can be float or array
    if np.any(eta_e > 1):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


def SNe_energy_gain(eta_e, mstar_dot):
    """energy associated with SN-powered galactic winds,"""
    norm = 100  # per 100 solar masses
    e_perSNE = 1e51 * u.erg
    return (eta_e * (mstar_dot) * (e_perSNE / (norm * u.solMass))).to(u.erg * u.Gyr**-1)


def dotE_SNe_wind(dot_m_star, M_halo):
    eta_E = vcirc_energy_loading(
        circular_velocity(M_halo, virial_radius(z, M_halo))
    )  # energy loading factor
    dot_e_ism_wind_erg_per_Gyr = SNe_energy_gain(eta_E, dot_m_star).to(u.erg / u.Gyr)
    return dot_e_ism_wind_erg_per_Gyr


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


# def dotE_CGM_acc(mhalo, Tvir):
#     """Energy gain of the CGM from accretion onto the halo

#     Args:
#         mhalo (_type_): halo mass
#         z (_type_): redshift
#     """
#     mdot_h = mdot_halo(z, mhalo)
#     dot_m_cgm_in = f_prevent * f_baryon * mdot_h  # baryonic accretion rate onto CGM
#     # print(dot_m_cgm_in)
#     kb = consts.k_B
#     mp = consts.m_p
#     mu = 0.6
#     return (kb / (mu * mp) * Tvir * dot_m_cgm_in).to(u.erg / u.Gyr)


def dotE_CGM_in_v1(mhalo, T_CGM, z, mcgm_hot, dotE_CGM_ej):
    kb = consts.k_B
    mp = consts.m_p
    mu = 3 / 5
    prefac = (kb / (mu * mp)) ** 2
    mdot_h = mdot_halo(z, mhalo)
    rvir = virial_radius(z, mhalo)
    T_vir = virial_T(mhalo, rvir)
    dotE = (prefac * T_vir**2 * f_baryon**2 * mdot_h**2) / dotE_CGM_ej
    return dotE.to(u.erg / u.Gyr)


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


def density0(mCGM_hot, r0, Rvir, alpha=1.4):
    """assuming rho = rho0 * (r/r1)^-alpha, this function computes rho0"""
    rho_0_hot = (mCGM_hot * (3 - alpha)) / (
        (4 * np.pi * r0 ** (3) * ((Rvir / r0) ** (3 - alpha) - 1))
    )

    return rho_0_hot.to(u.g / u.cm**3)


def cooling_rate_CGM(m_cgm_hot, r0, Rvir, T_CGM, metallicity, alpha=1.4):
    mu = 0.6 * consts.m_p  # mean molecular weight
    rho_0_hot = density0(m_cgm_hot, r0, Rvir, alpha=alpha).to(u.g / u.cm**3)
    lambda_cool = cooling_fn.cooling_function(T_CGM.value, metallicity) * (
        u.erg * u.cm**3 / u.s
    )  # in erg cm^3 / s
    # print(rho_0_hot, lambda_cool)
    frac = (4 * np.pi * rho_0_hot**2 * (r0.to(u.cm)) ** 3) / (mu**2 * (3 - 2 * alpha))
    pwr = 3 - 2 * alpha
    dot_e_cgm_cool = frac * lambda_cool * ((Rvir.to(u.cm) / r0.to(u.cm)) ** pwr - 1)
    return dot_e_cgm_cool.to(u.erg / u.Gyr)


def sfr_derived(m_ism, rvir, kappa_sfr=0.1, n_sfr=1.4):
    rvir = rvir.value  # in kpc
    m_ism = m_ism.to(u.Msun).value  # in msun
    r_disk = 0.02 * rvir
    sigma_gas = m_ism / (2 * np.pi * r_disk**2)  # msun / kpc^2

    A_SFR = 1e-3 * kappa_sfr
    mdot_msun_per_gyr = A_SFR * sigma_gas**n_sfr * (2 * np.pi * r_disk**2) / n_sfr**2
    # print(sigma_gas)
    return mdot_msun_per_gyr


def dotE_CGM_ej(m_cgm_hot, halo_rvir_kpc, mhalo, T_CGM, T_vir):
    mu = 0.6  # mean molecular weight
    c_sound = np.sqrt(consts.k_B * T_CGM / (consts.m_p * mu)).to(u.km / u.s)
    t_eject = (halo_rvir_kpc / c_sound).to(u.Gyr)

    # clip teject to 0.1 and 1 of t_ff
    t_ff_vir = t_ff(halo_rvir_kpc, halo_rvir_kpc, mhalo).to(u.Gyr)

    t_eject = np.clip(t_eject, 0.1 * t_ff_vir, 1.0 * t_ff_vir)

    dotE = consts.k_B / (mu * consts.m_p) * (m_cgm_hot / t_eject) * (T_CGM - T_vir)

    if dotE < 0:
        return 0 * u.erg / u.Gyr

    return dotE.to(u.erg / u.Gyr)


def dotE_CGM_ej_simplified(m_cgm_hot, T_CGM, T_vir, mhalo):
    mu = 0.6
    kb = consts.k_B
    mp = consts.m_p
    rvir = virial_radius(z, mhalo).to(u.kpc)

    dotE = (
        (kb / (mu * mp)) ** (3 / 2) * (m_cgm_hot / rvir) * (T_CGM - T_vir) * T_CGM**0.5
    )

    # if dotE < 0:
    #     dotE = 0 * u.erg / u.Gyr

    return dotE.to(u.erg / u.Gyr)


def dotE_CGM_in(mhalo, T_CGM, T_vir, z, mcgm_hot):
    kb = consts.k_B
    mp = consts.m_p
    mu = 3 / 5
    prefac = (kb / (mu * mp)) ** 0.5
    mdot_h = mdot_halo(z, mhalo).to(u.Msun / u.Gyr)
    rvir = virial_radius(z, mhalo).to(u.kpc)
   
    
    # print(mdot_h, rvir, T_vir)
    
    temp_terms = T_vir**2 / ((T_CGM - T_vir) * T_CGM**0.5)
   
    
    
    # apply the same units to test as temp_terms


    # if temp_terms < 0:
    #     temp_terms = 0 * temp_terms

    dot_e_cgm_in = prefac * temp_terms * f_baryon**2 * mdot_h**2 * (rvir / mcgm_hot)

    # if dot_e_cgm_in < 0:
    #     dot_e_cgm_in = 0 * u.erg / u.Gyr
    # print(dot_e_cgm_in)
    return dot_e_cgm_in.to(u.erg / u.Gyr)


def calculate_dotE_grid(
    n,
    m_cgm_hot_range,
    T_CGM_range,
    dot_mstar_derived,
    m_halo,
    halo_Tvir_K,
    halo_rvir_kpc,
    r0,
    metallicity,
    alpha,
):
    """Calculate dotE_CGM grid for parameter ranges"""
    
    M_grid, T_grid = np.meshgrid(m_cgm_hot_range, T_CGM_range)
    dotE_grid = np.zeros((n, n))

    dot_SNe_grid = np.zeros((n, n))
    dot_ej_grid = np.zeros((n, n))
    dotE_acc_grid = np.zeros((n, n))
    dotE_cool_grid = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dotE_SNe = dotE_SNe_wind(dot_mstar_derived, m_halo)
            dotE_ej = dotE_CGM_ej_simplified(
                10 ** M_grid[i, j] * u.Msun,
                10 ** T_grid[i, j] * u.K,
                halo_Tvir_K,
                m_halo,
            )
            # dotE_ej = dotE_CGM_ej(
            #     10 ** M_grid[i, j] * u.Msun,
            #     halo_rvir_kpc,
            #     m_halo,
            #     10 ** T_grid[i, j] * u.K,
            #     halo_Tvir_K,
            # )
            dotE_acc = dotE_CGM_in(
                m_halo, 10 ** T_grid[i, j] * u.K, halo_Tvir_K, z, 10 ** M_grid[i, j] * u.Msun
            )
            dotE_cool = cooling_rate_CGM(
                10 ** M_grid[i, j] * u.Msun,
                r0,
                halo_rvir_kpc,
                10 ** T_grid[i, j] * u.K,
                metallicity,
                alpha=alpha,
            )

            # print the units
            # print(dotE_SNe.unit, dotE_acc.unit, dotE_cool.unit, dotE_ej.unit)

            dotE_grid[i, j] = (dotE_SNe + dotE_acc - dotE_cool - dotE_ej).value
            dot_SNe_grid[i, j] = dotE_SNe.value
            dot_ej_grid[i, j] = dotE_ej.value
            dotE_acc_grid[i, j] = dotE_acc.value
            dotE_cool_grid[i, j] = dotE_cool.value

    return (
        M_grid,
        T_grid,
        dotE_grid,
        dot_SNe_grid,
        dot_ej_grid,
        dotE_acc_grid,
        dotE_cool_grid,
    )


### fixed properties of our analysis
alpha = 1.4  # density profile slope

z = 10  # redshift of analysis
# f_prevent = 1
m_halo = 5e9 * u.Msun
metallicity = 0.1  # in solar units
dot_m_star = 2e8 * u.Msun / u.Gyr

m_cgm_hot = 1e12 * u.Msun  # assume half baryons in hot CGM
T_CGM = 1e9 * u.K  # fixed CGM temperature


### derived halo properties
m_ism = 1e9 * u.Msun  # ISM mass
halo_rvir_kpc = virial_radius(z, m_halo).to(u.kpc)
r0 = 0.1 * halo_rvir_kpc
halo_vcirc_kms = circular_velocity(m_halo, halo_rvir_kpc)
halo_Tvir_K = virial_T(m_halo, halo_rvir_kpc).to(u.K)
dot_mstar_derived = (
    sfr_derived(m_ism, halo_rvir_kpc, kappa_sfr=0.1, n_sfr=1.4) * u.Msun / u.Gyr
)

### main energy rates
# dotE_SNe_wind_value = dotE_SNe_wind(dot_m_star, m_halo)  # in erg/Gyr
# dotE_CGM_in_value = dotE_CGM_in(m_halo, T_CGM, z, m_cgm_hot)  # in erg/Gyr
# # dotE_CGM_ej_value = dotE_CGM_ej(
# #     m_cgm_hot, halo_rvir_kpc, m_halo, T_CGM, halo_Tvir_K
# # )  # in erg/Gyr
# dotE_CGM_ej_value = dotE_CGM_ej_simplified(
#     m_cgm_hot, T_CGM, T_vir=halo_Tvir_K, mhalo=m_halo
# )
# dotE_CGM_cool_value = cooling_rate_CGM(
#     m_cgm_hot, r0, halo_rvir_kpc, T_CGM, metallicity, alpha=alpha
# )  # in erg/Gyr
# dotE_CGM = (
#     dotE_SNe_wind_value + dotE_CGM_in_value - dotE_CGM_cool_value - dotE_CGM_ej_value
# )


# plot the dotE grid without f_prevent dependence
n = 128
m_cgm_hot_range = np.linspace(7, 11, n)
T_CGM_range = np.linspace(4.6, 7.5, n)
M_grid, T_grid, dotE_grid, dot_SNe_grid, dot_ej_grid, dotE_acc_grid, dotE_cool = (
    calculate_dotE_grid(
        n,
        m_cgm_hot_range,
        T_CGM_range,
        dot_mstar_derived,
        m_halo,
        halo_Tvir_K,
        halo_rvir_kpc,
        r0,
        metallicity,
        alpha,
    )
)


# text for the params
textstr = ",\t".join(
    (
        r"$z={}$".format(z),
        r"$M_{{\rm halo}}={:.2e} {{\rm M_\odot}}$".format(m_halo.value),
        # r"$M_{{\rm CGM, hot}}={:.2e} ~ {{\rm M_\odot}}$".format(m_cgm_hot.value) ,
        r"$\dot{{M}}_\star={:.2e} ~ {{\rm M_\odot \:Gyr^{{-1}}}}$".format(
            dot_m_star.value
        ),
        r"$Z={:.3f} ~{{Z_\odot}}$".format(metallicity),
        r"$\alpha={}$".format(f"{alpha}"),
        # r"$f_{{\mathrm{{prevent}}}}={}$".format(f"{f_prevent}"),
    )
)

vmax = max(abs(dotE_grid.max()), abs(dotE_grid.min()))
linthresh = vmax * 1e-7
norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
im = ax.imshow(
    dotE_grid.T,
    extent=[
        T_CGM_range[0],
        T_CGM_range[-1],
        m_cgm_hot_range[0],
        m_cgm_hot_range[-1],
    ],
    origin="lower",
    aspect="auto",
    cmap="cmr.fusion_r",
    norm=norm,
    interpolation="nearest",
)

ax.set_xlabel(r"$\log T_{\mathrm{CGM}}~ {\rm [K]}$")
ax.set_ylabel(r"$\log M_{\mathrm{CGM, hot}}~ {\rm [M_\odot]}$")
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.text(
    0.0,
    1.05,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)
cbar_ax = ax.inset_axes([1.01, 0.0, 0.03, 1])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label(r"$\dot{E}_{\mathrm{CGM}} \: [{\rm erg\:Gyr^{-1}}]$")

# color bars
desired_ticks = 9
locator = SymmetricalLogLocator(linthresh=linthresh, base=10)
locator.set_params(numticks=desired_ticks)  # set number of ticks *after* creation
cbar.locator = locator
cbar.update_ticks()

cbar.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:g}")
cbar.update_ticks()

# less clutter by removing ticks close to zero
ticks = cbar.get_ticks()
nonzero = ticks[ticks != 0]
sorted_nonzero = nonzero[np.argsort(np.abs(nonzero))]
to_remove = sorted_nonzero[:2]
filtered = [t for t in ticks if t not in to_remove]
cbar.set_ticks(filtered)
cbar.update_ticks()

plt.show()

# %% PAPER FIGURE

mhalo_z0 = 1e11 * u.Msun
t_span = (0.1, 13.3)  # gyrs
model_ = CGMRegulator(
    mhalo_z0,
    (0.15, 13),
    add_f_prevent_floor=1e-6,
    verbose=False,
    # alpha_e=0.1,
    # alpha_m=0.1,
    updated_loadings=True,
    updated_halo_infall=True,
)
run_ = model_.run_halo()
results_ = model_.get_results()
derived_ = model_.get_derived_quantities()
redshift = results_["z"]  ##
time_gyr = results_["t"]

mstar_actual = results_["m_star"]
dot_mstar_actual = derived_["dot_m_sfr"]  ## needed for map
mism_actual = results_["m_ism"]
# dot_

m_cgm_hot_actual = results_["m_cgm_hot"]  ##
m_cgm_cold_actual = results_["m_cgm_cold"]
m_cgm_total_actual = m_cgm_hot_actual + m_cgm_cold_actual

e_cgm_total_actual = results_["egy_cgm"]  ##

# get rates for cgm masses
dot_m_cgm_hot_actual = derived_["dot_m_cgm_hot"]
dot_m_cgm_cold_actual = derived_["dot_m_cgm_cold"]

# get halo mass
mhalo_actual = results_["m_halo"]  ##

# get temperatures
Tcgm_actual = derived_["cgm_temp"]  ##
T_vir_actual = derived_["halo_vir_temp"]  ##

f_prevent_actual = derived_["f_prevent"]  ##
metallicity_actual = results_["metal_cgm_mass_sol"]  ##

dot_e_cgm_actual = (
    derived_["dot_e_cgm_in"]
    - derived_["dot_e_cgm_cooling"]
    - derived_["dot_e_cgm_out"]
    + derived_["dot_e_ism_wind"]
)

# %%

z2s_to_try = [15,12,11,10,9,6,7,6,5,2, 1, 0.8, 0.5, 0.3, 0.2, 0.1]
n_inset = 64
lin_threshold_factor = 1e-10
z1 = 9  # high z
# z2 = 0.3
for i, z2 in enumerate(z2s_to_try):
    fig, ax = plt.subplots(
        3,
        2,
        figsize=(10, 6),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.05)

    # add an inset for the equilibrium analysis to the right

    inset_hiz = ax[0].inset_axes([0, -4.45, 1.0, 1.5])
    cbar_ax_hiz = ax[0].inset_axes([0, -2.85, 1, 0.1])

    inset_loz = ax[1].inset_axes([0, -4.45, 1.0, 1.5])
    cbar_ax = ax[1].inset_axes([0, -2.85, 1, 0.1])

    star_color = "darkorange"
    ism_color = "tab:blue"
    cgm_cold_color = "dodgerblue"
    cgm_hot_color = "crimson"
    lw = 2
    # plot the masses of the ISM and CGM and stars over time
    ax[0].plot(time_gyr, mstar_actual, color=star_color, label=r"$M_\star$", lw=lw)
    ax[0].plot(
        time_gyr,
        m_cgm_cold_actual,
        color=cgm_cold_color,
        label=r"$M_{\mathrm{CGM, cold}}$",
        lw=lw,
    )
    ax[0].plot(
        time_gyr,
        m_cgm_hot_actual,
        color=cgm_hot_color,
        label=r"$M_{\mathrm{CGM, hot}}$",
        lw=lw,
    )

    ax[0].set(
        xscale="log",
        yscale="log",
        # xlabel=r"Cosmic time $t$ [Gyr]",
        ylabel=r"Mass [M$_{\odot}$]",
        ylim=(1e5, None),
    )

    # add the energy over time
    ax[2].plot(
        time_gyr,
        e_cgm_total_actual,
        color="tab:purple",
        lw=lw,
    )
    ax[2].set(
        xscale="log",
        yscale="log",
        ylabel=r"$E_{\mathrm{CGM}}$ [erg]",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        ylim=(1e53, None),
    )

    ax[3].plot(
        time_gyr,
        dot_e_cgm_actual,
        color="tab:brown",
        lw=lw,
    )
    ax[3].plot(
        time_gyr, -dot_e_cgm_actual, color="tab:brown", lw=lw, ls="--", alpha=0.2
    )
    ax[3].set(
        xscale="log",
        yscale="log",
        ylabel=r"$\dot{E}_{\mathrm{CGM}}$" + r" [erg Gyr$^{-1}$]",
        xlim=(0.14, 13),
        ylim=(1e53, None),
    )
    # ax[1].legend(frameon=False, fontsize=9, loc="lower right")

    # now do the rates for the masses above
    ax[1].plot(
        time_gyr, dot_mstar_actual, color=star_color, label=r"$\dot{M}_\star$", lw=lw
    )

    ax[1].plot(
        time_gyr,
        dot_m_cgm_cold_actual,
        color=cgm_cold_color,
        label=r"$\dot{M}_{\mathrm{CGM, cold}}$",
        lw=lw,
    )
    ax[1].plot(
        time_gyr,
        dot_m_cgm_hot_actual,
        color=cgm_hot_color,
        label=r"$\dot{M}_{\mathrm{CGM, hot}}$",
        lw=lw,
    )

    # add the negative values as dashed lines
    ax[1].plot(
        time_gyr,
        -dot_m_cgm_cold_actual,
        color=cgm_cold_color,
        lw=lw,
        ls="--",
        alpha=0.2,
    )
    ax[1].plot(
        time_gyr,
        -dot_m_cgm_hot_actual,
        color=cgm_hot_color,
        lw=lw,
        ls="--",
        alpha=0.2,
    )
    ax[1].plot(
        time_gyr,
        -dot_mstar_actual,
        color=star_color,
        lw=lw,
        ls="--",
        alpha=0.2,
    )

    ax[1].set(
        xscale="log",
        yscale="log",
        ylabel=r"mass rate" " [M$_{\odot}$ Gyr$^{-1}$]",
        xlim=(0.14, 13),
        ylim=(1e5, None),
    )

    # plot evolution of f_prevent and metallicity in the next two panels
    ax[4].plot(time_gyr, f_prevent_actual, color="tab:green", lw=lw)
    ax[4].set(
        xscale="log",
        ylabel=r"$f_{\mathrm{prevent}}$",
        ylim=(2e-3, 2),
        yscale="log",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
    )
    # ax[4].plot(time_gyr, metallicity_actual, color="tab:purple", lw=lw)
    # ax[4].set(
    #     xscale="log",
    #     yscale="log",
    #     ylabel=r"$Z_{\mathrm{CGM}}$ [$Z_{\odot}$]",
    #     ylim=(1e-3, 1),
    #     xlabel=r"$t_{\rm univ}$ [Gyr]",
    # )

    # then temp on the bottom row
    ax[5].plot(
        time_gyr, Tcgm_actual, color="tab:orange", lw=lw, label=r"$T_{\mathrm{CGM}}$"
    )
    ax[5].plot(
        time_gyr, T_vir_actual, color="tab:blue", lw=lw, label=r"$T_{\mathrm{vir}}$"
    )
    ax[5].set(
        xscale="log",
        yscale="log",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        ylabel=r"$T$ [K]",
        ylim=(5e4, 2e7),
    )
    ax[5].legend(ncols=2, frameon=False, fontsize=9, loc="lower right")

    t_ticks = np.array([0.2, 0.3, 0.5, 1, 2, 3, 5, 10])
    z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
    ax2 = ax[0].twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    ax2.set_xlabel(r"$z$", labelpad=8)
    ax2.minorticks_off()
    ax[0].minorticks_on()

    axz1 = ax[1].twiny()
    axz1.set_xscale("log")
    axz1.set_xlim(ax[1].get_xlim())
    axz1.set_xticks(t_ticks)
    axz1.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    axz1.set_xlabel(r"$z$", labelpad=8)
    ax[1].minorticks_on()
    axz1.minorticks_off()

    for axes in ax:
        for line in axes.lines:
            line.set_zorder(1)

    ax[0].legend(frameon=False, ncols=3, fontsize=10, loc="lower center")

    ########## equilibrium analysis ##########

    idx_z1 = np.argmin(np.abs(redshift - z1))
    f_prevent_z1 = f_prevent_actual[idx_z1]
    m_ism_z1 = mism_actual[idx_z1] * u.Msun
    m_halo_z1 = mhalo_actual[idx_z1] * u.Msun
    m_cgm_hot_z1 = m_cgm_hot_actual[idx_z1] * u.Msun
    T_CGM_z1 = Tcgm_actual[idx_z1] * u.K
    metallicity_z1 = metallicity_actual[idx_z1]
    dot_m_star_z1 = dot_mstar_actual[idx_z1] * u.Msun / u.Gyr
    halo_rvir_kpc_z1 = virial_radius(z1, m_halo_z1).to(u.kpc)
    halo_Tvir_K_z1 = T_vir_actual[idx_z1] * u.K
    r0_z1 = 0.1 * halo_rvir_kpc_z1

    # now repeat for z2
    idx_z2 = np.argmin(np.abs(redshift - z2))
    f_prevent_z2 = f_prevent_actual[idx_z2]
    m_ism_z2 = mism_actual[idx_z2] * u.Msun
    m_halo_z2 = mhalo_actual[idx_z2] * u.Msun
    m_cgm_hot_z2 = m_cgm_hot_actual[idx_z2] * u.Msun
    T_CGM_z2 = Tcgm_actual[idx_z2] * u.K
    metallicity_z2 = metallicity_actual[idx_z2]
    dot_m_star_z2 = dot_mstar_actual[idx_z2] * u.Msun / u.Gyr
    dotE_actual_z2 = dot_e_cgm_actual[idx_z2]
    halo_rvir_kpc_z2 = virial_radius(z2, m_halo_z2).to(u.kpc)
    halo_Tvir_K_z2 = T_vir_actual[idx_z2] * u.K
    r0_z2 = 0.1 * halo_rvir_kpc_z2

    # add vertical line for z1
    for axes in ax:
        axes.axvline(
            x=time_gyr[idx_z1],
            color="gray",
            ls="-",
            lw=2,
            alpha=0.7,
            zorder=10,
        )
        axes.axvline(
            x=time_gyr[idx_z2],
            color="gray",
            ls=":",
            lw=2,
            alpha=0.7,
            zorder=10,
        )

    m_cgm_hot_range_inset = np.linspace(4.9, np.log10(m_halo_z2.value), n_inset)
    m_cgm_hot_range_inset_z2 = np.linspace(4.9, np.log10(m_halo_z2.value), n_inset)
    T_CGM_range_inset = np.linspace(5.5, 7.5, n_inset)
    T_CGM_range_inset_z2 = np.linspace(5.5, 7.5, n_inset)
  

    ######
    (
        M_grid_inset,
        T_grid_inset,
        dotE_grid_inset,
        dot_SNe_grid_inset,
        dot_ej_grid_inset,
        dotE_acc_grid_inset,
        dotE_cool_grid_inset,
    ) = calculate_dotE_grid(
        n_inset,
        m_cgm_hot_range_inset,
        T_CGM_range_inset,
        dot_m_star_z1,  ##
        m_halo_z1,
        halo_Tvir_K_z1,
        halo_rvir_kpc_z1,
        r0_z1,
        metallicity_z1,
        alpha,
    )
    (
        M_grid_inset_z2,
        T_grid_inset_z2,
        dotE_grid_inset_z2,
        dot_SNe_grid_inset_z2,
        dot_ej_grid_inset_z2,
        dotE_acc_grid_inset_z2,
        dotE_cool_grid_inset_z2,
    ) = calculate_dotE_grid(
        n_inset,
        m_cgm_hot_range_inset_z2,
        T_CGM_range_inset_z2,
        dot_m_star_z2,
        m_halo_z2,
        halo_Tvir_K_z2,
        halo_rvir_kpc_z2,
        r0_z2,
        metallicity_z2,
        alpha,
    )

    dot_e_cgm_actual_z1 = dot_e_cgm_actual[idx_z1]
    # normalize to absolute value for plotting
    dotE_grid_inset_normalized = dotE_grid_inset / abs(dot_e_cgm_actual_z1)
    dotE_grid_inset_normalized_z2 = dotE_grid_inset_z2 / abs(dotE_actual_z2)
    # color bar for inset
    vmax = max(
        abs(dotE_grid_inset_normalized.max()), abs(dotE_grid_inset_normalized.min())
    )
    linthresh = vmax * lin_threshold_factor
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

    vmax2 = max(
        abs(dotE_grid_inset_normalized_z2.max()),
        abs(dotE_grid_inset_normalized_z2.min()),
    )
    norm_z2 = SymLogNorm(linthresh=linthresh, vmin=-vmax2, vmax=vmax2)
    #####

    im_inset = inset_hiz.imshow(
        dotE_grid_inset_normalized.T,
        extent=[
            T_CGM_range_inset[0],
            T_CGM_range_inset[-1],
            m_cgm_hot_range_inset[0],
            m_cgm_hot_range_inset[-1],
        ],
        origin="lower",
        aspect="auto",
        cmap="cmr.fusion_r",
        norm=norm,
        interpolation="nearest",
    )
    inset_hiz.set(
        # xlabel=r"$\log T_{\mathrm{CGM}}~ {\rm [K]}$",
        ylabel=r"$\log M_{\mathrm{CGM, hot}}~ {\rm [M_\odot]}$",
    )

    cbar_inset = plt.colorbar(im_inset, cax=cbar_ax, orientation="horizontal")
    cbar_inset.set_label(
        r"$\dot{E}_{\mathrm{CGM}} / |\dot{E}_{\mathrm{CGM}}(T_{\rm CGM}, M_{\rm CGM, hot})| \: [{\rm erg\:Gyr^{-1}}]$"
    )

    # color bars
    desired_ticks = 9
    locator = SymmetricalLogLocator(linthresh=linthresh, base=10)
    locator.set_params(numticks=desired_ticks)  # set number of ticks *after* creation
    cbar_inset.locator = locator
    cbar_inset.update_ticks()
    cbar_inset.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:g}")
    cbar_inset.update_ticks()

    # less clutter by reoving ticks close to zero
    ticks = cbar_inset.get_ticks()
    nonzero = ticks[ticks != 0]
    sorted_nonzero = nonzero[np.argsort(np.abs(nonzero))]
    to_remove = sorted_nonzero[:2]
    filtered = [t for t in ticks if t not in to_remove]
    cbar_inset.set_ticks(filtered)
    cbar_inset.update_ticks()

    # mark the current state in the inset
    inset_hiz.axvline(
        np.log10(T_CGM_z1.value),
        color="black",
        ls="-",
        lw=2,
        alpha=0.2,
        zorder=10,
    )
    inset_hiz.axhline(
        np.log10(m_cgm_hot_z1.value),
        color="black",
        ls="-",
        lw=2,
        alpha=0.2,
        zorder=10,
    )

    ##### add another for z2, and use same color bar
    im_inset_z2 = inset_loz.imshow(
        dotE_grid_inset_normalized_z2.T,
        extent=[
            T_CGM_range_inset_z2[0],
            T_CGM_range_inset_z2[-1],
            m_cgm_hot_range_inset_z2[0],
            m_cgm_hot_range_inset_z2[-1],
        ],
        origin="lower",
        aspect="auto",
        cmap="cmr.fusion_r",
        norm=norm_z2,
        interpolation="nearest",
    )

    # second color bar
    cbar_highz = plt.colorbar(im_inset_z2, cax=cbar_ax_hiz, orientation="horizontal")
    cbar_highz.set_label(
        r"$\dot{E}_{\mathrm{CGM}} / |\dot{E}_{\mathrm{CGM}}(T_{\rm CGM}, M_{\rm CGM, hot})| \: [{\rm erg\:Gyr^{-1}}]$"
    )

    desired_ticks = 9
    locator = SymmetricalLogLocator(linthresh=linthresh, base=10)
    locator.set_params(numticks=desired_ticks)  # set number of ticks *after* creation
    cbar_highz.locator = locator
    cbar_highz.update_ticks()
    cbar_highz.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:g}")
    cbar_highz.update_ticks()
    # less clutter by removing ticks close to zero
    ticks = cbar_highz.get_ticks()
    nonzero = ticks[ticks != 0]
    sorted_nonzero = nonzero[np.argsort(np.abs(nonzero))]
    to_remove = sorted_nonzero[:2]
    filtered = [t for t in ticks if t not in to_remove]
    cbar_highz.set_ticks(filtered)
    cbar_highz.update_ticks()

    # set tick and label to uptop
    cbar_highz.ax.xaxis.set_ticks_position("top")
    cbar_highz.ax.xaxis.set_label_position("top")

    cbar_inset.ax.xaxis.set_ticks_position("top")
    cbar_inset.ax.xaxis.set_label_position("top")

    inset_loz.set(
        xlabel=r"$\log T_{\mathrm{CGM}}~ {\rm [K]}$",
        ylabel=r"$\log M_{\mathrm{CGM, hot}}~ {\rm [M_\odot]}$",
    )

    inset_loz.axvline(
        np.log10(T_CGM_z2.value),
        color="black",
        ls=":",
        lw=2,
        alpha=0.2,
        zorder=10,
    )
    inset_loz.axhline(
        np.log10(m_cgm_hot_z2.value),
        color="black",
        ls=":",
        lw=2,
        alpha=0.2,
        zorder=10,
    )

    # add text for z1 inside inset
    z1_textstr = ",\t".join(
        (
            r"$z={}$".format(z1),
            r"$M_{{\rm halo}}={} {{\rm M_\odot}}$".format(
                format_sci_notation(m_halo_z1.value, 1)
            ),
            # r"$\dot{{M}}_\star={} ~ {{\rm M_\odot \:Gyr^{{-1}}}}$".format(
            #     format_sci_notation(dot_m_star_z1.value, 1)
            # ),
            # r"$Z={:.3f} ~{{Z_\odot}}$".format(metallicity_z1),
            # r"$f_{{\mathrm{{prevent}}}}={:.2f}$".format(f_prevent_z1),
        )
    )
    inset_hiz.text(
        0.05,
        0.95,
        z1_textstr,
        transform=inset_hiz.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
    )

    z2_textstr = ",\t".join(
        (
            r"$z={}$".format(z2),
            r"$M_{{\rm halo}}={} {{\rm M_\odot}}$".format(
                format_sci_notation(m_halo_z2.value, 1)
            ),
            # r"$\dot{{M}}_\star={} ~ {{\rm M_\odot \:Gyr^{{-1}}}}$".format(
            #     format_sci_notation(dot_m_star_z2.value, 1)
            # ),
            # r"$Z={:.3f} ~{{Z_\odot}}$".format(metallicity_z2),
            # r"$f_{{\mathrm{{prevent}}}}={:.2f}$".format(f_prevent_z2),
        )
    )
    inset_loz.text(
        0.05,
        0.95,
        z2_textstr,
        transform=inset_loz.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
    )

    # annotate the z= 0 halo mass on the top panel
    ax[0].text(
        0.05,
        0.95,
        r"$M_{{\rm halo}} (z=0) = {} \: {{\rm M_\odot}}$".format(
            format_sci_notation(results_["m_halo"][-1], decimals=0)
        ),
        fontsize=10,
        color="black",
        transform=ax[0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    savetext = f"./figures/equilibrium_scan_1e11/{i:02d}.png"
    plt.savefig(savetext, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.show()

# make a plot of all the relavant rates

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def plot_phase_grid_2x5_from_grids_with_actual(
    grids_z1,
    grids_z2,
    extent,
    Mcgm_hot_actual_z1,
    Tcgm_actual_z1,
    Mcgm_hot_actual_z2,
    Tcgm_actual_z2,
    z1,
    z2,
    linthresh_prefactor=1e-10,
    cmap="coolwarm",
    figsize=(15, 6),
    alpha=0.9,
    lw_actual=2.0,
):
    """
    2x5 phase-space grid with per-panel normalization and actual CGM tracks.

    grids_z1, grids_z2 : dict
        Keys:
            'dotE'
            'dot_SNe'
            'dot_ej'
            'dotE_acc'
            'dotE_cool'
        Values: 2D numpy arrays

    extent : (xmin, xmax, ymin, ymax)

    Mcgm_hot_actual_* : 1D array
        log M_cgm,hot trajectory

    Tcgm_actual_* : 1D array
        log T_cgm trajectory
    """

    keys = [
        "dotE",
        "dot_SNe",
        "dot_ej",
        "dotE_acc",
        "dotE_cool",
    ]

    titles = [
        r"$\dot{E}_{\rm CGM}$",
        r"$\dot{E}_{\rm SNe}$",
        r"$\dot{E}_{\rm ej}$",
        r"$\dot{E}_{\rm acc}$",
        r"$\dot{E}_{\rm cool}$",
    ]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=figsize,
        dpi=300,
    )
    for ax in axes[0, :]:
        ax.axhline(Mcgm_hot_actual_z1, color="k", ls="-", lw=1, alpha=0.75)
        ax.axvline(Tcgm_actual_z1, color="k", ls="-", lw=1, alpha=0.75)
    for ax in axes[1, :]:
        ax.axhline(Mcgm_hot_actual_z2, color="k", ls="-", lw=1, alpha=0.75)
        ax.axvline(Tcgm_actual_z2, color="k", ls="-", lw=1, alpha=0.75)
    for row, (grids, Mcgm, Tcgm) in enumerate(
        [
            (grids_z1, Mcgm_hot_actual_z1, Tcgm_actual_z1),
            (grids_z2, Mcgm_hot_actual_z2, Tcgm_actual_z2),
        ]
    ):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            grid = grids[key]

            vmax = np.nanmax(np.abs(grid))
            linthresh = linthresh_prefactor * vmax

            norm = SymLogNorm(
                linthresh=linthresh,
                vmin=-vmax,
                vmax=vmax,
                base=10,
            )

            im = ax.imshow(
                grid.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                alpha=alpha,
            )

            # ---- actual CGM trajectory overlay ----

            if row == 0:
                ax.set_title(titles[col], fontsize=13)

            if col == 0:
                ax.set_ylabel(
                    r"$\log M_{\rm CGM,hot}$"
                    + (
                        "\n(z = {})".format(z1) if row == 0 else "\n(z = {})".format(z2)
                    ),
                    fontsize=12,
                )

            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.ax.tick_params(labelsize=8)

    axes[-1, 2].set_xlabel(r"$\log T_{\rm CGM}$", fontsize=12)

    fig.tight_layout()
    return fig, axes


# %%
grids_z1 = dict(
    dotE=dotE_grid_inset,
    dot_SNe=dot_SNe_grid_inset,
    dot_ej=dot_ej_grid_inset,
    dotE_acc=dotE_acc_grid_inset,
    dotE_cool=dotE_cool_grid_inset,
)

grids_z2 = dict(
    dotE=dotE_grid_inset_z2,
    dot_SNe=dot_SNe_grid_inset_z2,
    dot_ej=dot_ej_grid_inset_z2,
    dotE_acc=dotE_acc_grid_inset_z2,
    dotE_cool=dotE_cool_grid_inset_z2,
)

fig, axes = plot_phase_grid_2x5_from_grids_with_actual(
    grids_z1=grids_z1,
    grids_z2=grids_z2,
    extent=(
        T_CGM_range_inset[0],
        T_CGM_range_inset[-1],
        m_cgm_hot_range_inset[0],
        m_cgm_hot_range_inset[-1],
    ),
    Mcgm_hot_actual_z1=np.log10(m_cgm_hot_z1.value),
    Tcgm_actual_z1=np.log10(T_CGM_z1.value),
    Mcgm_hot_actual_z2=np.log10(m_cgm_hot_z2.value),
    Tcgm_actual_z2=np.log10(T_CGM_z2.value),
    linthresh_prefactor=1e-8,
    z1=z1,
    z2=z2,
)

vert_line_temp_terms = T_vir_actual[idx_z2] ** 2 / (
    (Tcgm_actual[idx_z2] - T_vir_actual[idx_z2]) * Tcgm_actual[idx_z2] ** 0.5
)

axes[1, 2].axvline(np.log10(T_vir_actual[idx_z2]), ls="--", color="k") 

test = Tcgm_actual[idx_z2] * 1.6
axes[1,3].axvline(np.log10(T_vir_actual[idx_z2]), ls="--", color="k")
plt.show()
