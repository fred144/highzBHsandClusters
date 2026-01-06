# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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


def dotE_CGM_acc(mhalo, Tvir):
    """Energy gain of the CGM from accretion onto the halo

    Args:
        mhalo (_type_): halo mass
        z (_type_): redshift
    """
    mdot_h = mdot_halo(z, mhalo)
    dot_m_cgm_in = f_prevent * f_baryon * mdot_h  # baryonic accretion rate onto CGM
    # print(dot_m_cgm_in)
    kb = consts.k_B
    mp = consts.m_p
    mu = 0.6
    return (kb / (mu * mp) * Tvir * dot_m_cgm_in).to(u.erg / u.Gyr)


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


def dotE_CGM_ej(m_cgm_hot, halo_rvir_kpc, mhalo, T_CGM, T_vir):
    mu = 0.6  # mean molecular weight
    c_sound = np.sqrt(consts.k_B * T_CGM / (consts.m_p * mu)).to(u.km / u.s)
    t_eject = (halo_rvir_kpc / c_sound).to(u.Gyr)

    # clip teject to 0.1 and 1 of t_ff
    t_ff_vir = t_ff(halo_rvir_kpc, halo_rvir_kpc, mhalo).to(u.Gyr)

    t_eject = np.clip(t_eject, 0.1 * t_ff_vir, 1.0 * t_ff_vir)

    dotE = consts.k_B / (mu * consts.m_p) * (m_cgm_hot / t_eject) * (T_CGM - T_vir)
    return dotE.to(u.erg / u.Gyr)


def density0(mCGM_hot, r0, Rvir, alpha=1.4):
    """assuming rho = rho0 * (r/r1)^-alpha, this function computes rho0"""
    rho_0_hot = (mCGM_hot * (3 - alpha)) / (
        (4 * np.pi * r0 ** (3) * ((Rvir / r0) ** (3 - alpha) - 1))
    )

    return rho_0_hot.to(u.g / u.cm**3)


def cooling_rate_CGM(m_cgm_hot, Mhalo, r0, Rvir, T_CGM, metallicity, alpha=1.4):
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


### fixed properties of our analysis
z = 6  # redshift of analysis
f_prevent = 1
alpha = 1.4  # density profile slope
m_ism = 1e9 * u.Msun  # ISM mass
m_halo = 1e11 * u.Msun
m_cgm_hot = 1e12 * u.Msun  # assume half baryons in hot CGM
T_CGM = 1e9 * u.K  # fixed CGM temperature

metallicity = 0.1  # in solar units
dot_m_star = 1e7 * u.Msun / u.Gyr


### derived halo properties
halo_rvir_kpc = virial_radius(z, m_halo).to(u.kpc)
r0 = 0.1 * halo_rvir_kpc
halo_vcirc_kms = circular_velocity(m_halo, halo_rvir_kpc)
halo_Tvir_K = virial_T(m_halo, halo_rvir_kpc).to(u.K)
dot_mstar_derived = (
    sfr_derived(m_ism, halo_rvir_kpc, kappa_sfr=0.1, n_sfr=1.4) * u.Msun / u.Gyr
)

### main energy rates
dotE_SNe_wind_value = dotE_SNe_wind(dot_m_star, m_halo)  # in erg/Gyr
dotE_CGM_acc_value = dotE_CGM_acc(m_halo, halo_Tvir_K)  # in erg/Gyr
dotE_CGM_ej_value = dotE_CGM_ej(
    m_cgm_hot, halo_rvir_kpc, m_halo, T_CGM, halo_Tvir_K
)  # in erg/Gyr
dotE_CGM_cool_value = cooling_rate_CGM(
    m_cgm_hot, m_halo, r0, halo_rvir_kpc, T_CGM, metallicity, alpha=alpha
)  # in erg/Gyr
dotE_CGM = (
    dotE_SNe_wind_value + dotE_CGM_acc_value - dotE_CGM_cool_value - dotE_CGM_ej_value
)


# print(dotE_CGM)
# %% create parameter ranges
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

    for i in range(n):
        for j in range(n):
            dotE_SNe = dotE_SNe_wind(dot_mstar_derived, m_halo)
            dotE_acc = dotE_CGM_acc(m_halo, halo_Tvir_K)
            dotE_ej = dotE_CGM_ej(
                10 ** M_grid[i, j] * u.Msun,
                halo_rvir_kpc,
                m_halo,
                10 ** T_grid[i, j] * u.K,
                halo_Tvir_K,
            )
            dotE_cool = cooling_rate_CGM(
                10 ** M_grid[i, j] * u.Msun,
                m_halo,
                r0,
                halo_rvir_kpc,
                10 ** T_grid[i, j] * u.K,
                metallicity,
                alpha=alpha,
            )
            dotE_grid[i, j] = (dotE_SNe + dotE_acc - dotE_cool - dotE_ej).value

    return M_grid, T_grid, dotE_grid


n = 128
m_cgm_hot_range = np.linspace(7, np.log10(0.8 * m_halo.value), n)
T_CGM_range = np.linspace(4.6, 7, n)
M_grid, T_grid, dotE_grid = calculate_dotE_grid(
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
        r"$f_{{\mathrm{{prevent}}}}={}$".format(f"{f_prevent}"),
    )
)

vmax = max(abs(dotE_grid.max()), abs(dotE_grid.min()))
linthresh = vmax * 1e-5
norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
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


# def dot_E_cgm():
#     return dotE_SNe_wind + dotE_CGM_acc - dot_E_CGM_cool - dot_E_CGM_ej

# %% PAPER FIGURE

mhalo_z0 = 1e12 * u.Msun
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

# get rates for cgm masses
dot_m_cgm_hot_actual = derived_["dot_m_cgm_hot"]
dot_m_cgm_cold_actual = derived_["dot_m_cgm_cold"]

# get halo mass
mhalo_actual = results_["m_halo"]  ##

# get temperatures
Tcgm_actual = derived_["cgm_temp"]  ##

f_prevent_actual = derived_["f_prevent"]  ##
metallicity_actual = results_["metal_cgm_mass_sol"]  ##

dot_e_cgm_actual = (
    derived_["dot_e_cgm_in"]
    - derived_["dot_e_cgm_cooling"]
    - derived_["dot_e_cgm_out"]
    + derived_["dot_e_ism_wind"]
)  ##

# %%
fig, ax = plt.subplots(
    5,
    1,
    figsize=(5, 9),
    dpi=300,
    sharex=True,
    gridspec_kw={"height_ratios": [3, 3, 1.25, 1.25, 1.25]},
)
plt.subplots_adjust(hspace=0.05)

# add an inset for the equilibrium analysis to the right

inset_hiz = ax[0].inset_axes([1.15, -0.35, 1.0, 1.35])
inset_loz = ax[0].inset_axes([1.15, -2.0, 1.0, 1.35])
cbar_ax = ax[0].inset_axes([1.15, -2.35, 1, 0.1])


star_color = "darkorange"
ism_color = "tab:blue"
cgm_cold_color = "dodgerblue"
cgm_hot_color = "crimson"
lw = 2
# plot the masses of the ISM and CGM and stars over time
ax[0].plot(time_gyr, mstar_actual, color=star_color, label=r"$M_\star$", lw=lw)
# ax[0].plot(
#     time_gyr, mism_actual,
#     color=ism_color,
#     label=r"$M_{\mathrm{ISM}}$",
# )
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
    ylim=(1e5, 5e10),
)

# now do the rates for the masses above
ax[1].plot(
    time_gyr, dot_mstar_actual, color=star_color, label=r"$\dot{M}_\star$", lw=lw
)
# ax[1].plot(
#     time_gyr, derived_["dot_m_ism"],
#     color=ism_color,
#     label=r"$\dot{M}_{\mathrm{ISM}}$",
# )
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

ax[1].set(
    xscale="log",
    yscale="log",
    ylabel=r"mass rate [M$_{\odot}$ Gyr$^{-1}$]",
    xlim=(0.15, 13),
    ylim=(1e6, 2e10),
)


# plot evolution of f_prevent and metallicity in the next two panels
ax[2].plot(time_gyr, f_prevent_actual, color="tab:green", lw=lw)
ax[2].set(xscale="log", ylabel=r"$f_{\mathrm{prevent}}$", ylim=(2e-3, 2), yscale="log")
ax[3].plot(time_gyr, metallicity_actual, color="tab:purple", lw=lw)
ax[3].set(
    xscale="log",
    yscale="log",
    ylabel=r"$Z_{\mathrm{CGM}}$ [$Z_{\odot}$]",
    ylim=(1e-3, 1),
)

# then temp on the bottom row
ax[4].plot(time_gyr, Tcgm_actual, color="tab:orange", lw=lw)
ax[4].set(
    xscale="log",
    yscale="log",
    xlabel=r"$t$ [Gyr]",
    ylabel=r"$T_{\mathrm{CGM}}$ [K]",
    ylim=(5e4, 2e7),
)


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
ax[1].minorticks_on()

for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)

ax[0].legend(frameon=False, loc="lower right", ncols=3)


##### Equilibrium analysis
z1 = 10  # high z
z2 = 1.2
lin_threshhold_factor = 1e-5

idx_z1 = np.argmin(np.abs(redshift - z1))
f_prevent_z1 = f_prevent_actual[idx_z1]
m_ism_z1 = mism_actual[idx_z1] * u.Msun
m_halo_z1 = mhalo_actual[idx_z1] * u.Msun
m_cgm_hot_z1 = m_cgm_hot_actual[idx_z1] * u.Msun
T_CGM_z1 = Tcgm_actual[idx_z1] * u.K
metallicity_z1 = metallicity_actual[idx_z1]
dot_m_star_z1 = dot_mstar_actual[idx_z1] * u.Msun / u.Gyr


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

# now make inset equilibrium map at z1 using calculate_dotE_grid
n_inset = 128
m_cgm_hot_range_inset = np.linspace(7, 11, n_inset)
T_CGM_range_inset = np.linspace(4.6, 7, n_inset)

######
# M_grid_inset, T_grid_inset, dotE_grid_inset = calculate_dotE_grid(
#     n_inset, m_cgm_hot_range_inset, T_CGM_range_inset, dot_m_star_z1, m_halo_z1,
#     halo_Tvir_K, halo_rvir_kpc, r0, metallicity_z1, alpha
# )
#####

dot_e_cgm_actual_z1 = dot_e_cgm_actual[idx_z1]
# normalize to absolute value for plotting
dotE_grid_inset_normalized = dotE_grid_inset / abs(dot_e_cgm_actual_z1)


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
inset_hiz.set_xlabel(r"$\log T_{\mathrm{CGM}}~ {\rm [K]}$")
inset_hiz.set_ylabel(r"$\log M_{\mathrm{CGM, hot}}~ {\rm [M_\odot]}$")


# color bar for inset
vmax = min(abs(dotE_grid_inset_normalized.max()), abs(dotE_grid_inset_normalized.min()))


linthresh = vmax * lin_threshhold_factor
norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)


cbar_inset = plt.colorbar(im_inset, cax=cbar_ax, orientation="horizontal")
cbar_inset.set_label(r"$\dot{E}_{\mathrm{CGM}} \: [{\rm erg\:Gyr^{-1}}]$")
# set tick location up top and also label
# cbar_inset.ax.xaxis.set_ticks_position("top")
# cbar_inset.ax.xaxis.set_label_position("top")

# color bars
desired_ticks = 11
locator = SymmetricalLogLocator(linthresh=linthresh, base=10)
locator.set_params(numticks=desired_ticks)  # set number of ticks *after* creation
cbar_inset.locator = locator
cbar_inset.update_ticks()

cbar_inset.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:g}")
cbar_inset.update_ticks()

# less clutter by removing ticks close to zero
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

# add text for z1 inside inset
z1_textstr = "\n".join(
    (
        r"$z={}$".format(z1),
        r"$M_{{\rm halo}}={:.2e} {{\rm M_\odot}}$".format(m_halo_z1.value),
        r"$\dot{{M}}_\star={:.2e} ~ {{\rm M_\odot \:Gyr^{{-1}}}}$".format(
            dot_m_star_z1.value
        ),
        r"$Z={:.3f} ~{{Z_\odot}}$".format(metallicity_z1),
        r"$\alpha={}$".format(f"{alpha}"),
        r"$f_{{\mathrm{{prevent}}}}={:.2f}$".format(f_prevent_z1),
    )
)
inset_hiz.text(
    0.05,
    0.05,
    z1_textstr,
    transform=inset_hiz.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    # bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)

##### add another for z2, and use same color bar

idx_z2 = np.argmin(np.abs(redshift - z2))

f_prevent_z2 = f_prevent_actual[idx_z2]
m_ism_z2 = mism_actual[idx_z2] * u.Msun
m_halo_z2 = mhalo_actual[idx_z2] * u.Msun
m_cgm_hot_z2 = m_cgm_hot_actual[idx_z2] * u.Msun
T_CGM_z2 = Tcgm_actual[idx_z2] * u.K
metallicity_z2 = metallicity_actual[idx_z2]
dot_m_star_z2 = dot_mstar_actual[idx_z2] * u.Msun / u.Gyr
dotE_actual_z2 = dot_e_cgm_actual[idx_z2]


n_inset = 128
m_cgm_hot_range_inset_z2 = np.linspace(7, 11, n_inset)
T_CGM_range_inset_z2 = np.linspace(4.6, 7, n_inset)

# M_grid_inset_z2, T_grid_inset_z2, dotE_grid_inset_z2 = calculate_dotE_grid(
#     n_inset, m_cgm_hot_range_inset_z2, T_CGM_range_inset_z2, dot_m_star_z2, m_halo_z2,
#     halo_Tvir_K, halo_rvir_kpc, r0, metallicity_z2, alpha
# )
dotE_grid_inset_normalized_z2 = dotE_grid_inset_z2 / abs(dotE_actual_z2)

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
    norm=norm,
    interpolation="nearest",
)
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

# add vertical line for z2
for axes in ax:
    axes.axvline(
        x=time_gyr[idx_z2],
        color="gray",
        ls=":",
        lw=2,
        alpha=0.7,
        zorder=10,
    )

z2_textstr = "\n".join(
    (
        r"$z={}$".format(z2),
        r"$M_{{\rm halo}}={:.2e} {{\rm M_\odot}}$".format(m_halo_z2.value),
        r"$\dot{{M}}_\star={:.2e} ~ {{\rm M_\odot \:Gyr^{{-1}}}}$".format(
            dot_m_star_z2.value
        ),
        r"$Z={:.3f} ~{{Z_\odot}}$".format(metallicity_z2),
        r"$\alpha={}$".format(f"{alpha}"),
        r"$f_{{\mathrm{{prevent}}}}={:.2f}$".format(f_prevent_z2),
    )
)
inset_loz.text(
    0.05,
    0.05,
    z2_textstr,
    transform=inset_loz.transAxes,
    fontsize=9,
    verticalalignment="bottom",
)


plt.show()

# make a plot of all the relavant rates

# %%
