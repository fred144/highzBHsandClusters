# %%
import os
from pyexpat import model
import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp
from astropy import cosmology
import h5py
import astropy.constants as consts
import astropy.units as u
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table, join, hstack, vstack
from cgm_sf_regulator import CGMRegulator, halo_diagnostics_v2
from cgm_sf_regulator_baseline import CGMRegulatorBaseline
from astropy.units import Quantity
from run_grids_of_models import (
    run_baseline_model_redshift_grid,
    run_2phase_model_redshift_grid,
)
import matplotlib.lines as mlines

### update matplotlib settings
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)
###

# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7

Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)

mass_bins = 16
# %% for z=0 comparisons

# read in behroozi data for comparison
smmr = np.loadtxt("./data/sm_averages_a1.002310.dat")
loghm = smmr[5:21, 0]  # halo mass
hm = 10**loghm * u.solMass

logSM = smmr[5:21, 7]  # stellar mass
sm = 10 ** logSM[5:21] * u.solMass

SMerr = np.vstack([smmr[5:21, 3], smmr[5:21, 2]])  # error array
logSMHM = smmr[5:21, 10]  # stellar mass / halo mass
SMHMerr = np.vstack([smmr[5:21, 12], smmr[5:21, 11]])  # error array

smhm_behroozi = 10**logSMHM / (Ob0 / Omegam0)
smhm_err_up = 10 ** SMHMerr[1, :]
smhm_err_low = 10 ** SMHMerr[0, :]


# define the redshift bins strings that will make it easier to read z-bins
zbins_str = [
    "0.2 < z < 0.5",
    # "0.5 < z < 0.8",
    # "0.8 < z < 1.1",
    # "1.1 < z < 1.5",
    # "1.5 < z < 2.0",
    "2.0 < z < 2.5",
    "2.5 < z < 3.0",
    # "3.0 < z < 3.5",
    "3.5 < z < 4.5",
    # "4.5 < z < 5.5",
    "5.5 < z < 6.5",
    "6.5 < z < 7.5",
    "7.5 < z < 8.5",
    # "8.5 < z < 10.0",
    "10.0 < z < 12.0",
]

# make a color map with the same number of colors as the number of zbins_str
cmap = cmr.tropical
cmap_colors = [cmap(i / len(zbins_str)) for i in range(len(zbins_str))]

zbins_ctr = []  # get the center value of zbins_str
for zb in zbins_str:
    z = zb.split("<")
    z = (float(z[0]) + float(z[2])) / 2
    zbins_ctr.append(z)

## read the table
smf_data = Table.read("./data/Shuntov2024-shmr.ecsv", format="ascii.ecsv")

# define empty dictionaries to store values for each z-bin
Mhalo = {}
Mstar = {}
Mstar_low = {}
Mstar_up = {}

# read for each z-bin
for zb in zbins_str:
    Mhalo[zb] = smf_data[smf_data["Redshift"] == zb]["M_halo"]
    Mstar[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_50"]
    Mstar_low[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_16"]
    Mstar_up[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_84"]

# reverse zbins_ctr to match the order of the data
zbins_ctr = zbins_ctr[::-1]
zbins_ctr.append(0.01)  # this is the lowest redshift bin


# %% halo properties
# zobs = [12, 11]  # redshifts to observe at
# zobs = [8, 7, 3, 2, 1, 0.05]  # redshifts to observe at

zobs = zbins_ctr


# # append ot the begining of zobs [20,15]
# add_these_zs = [20, 15]
# zobs = add_these_zs + zobs

# make a unique halo array for each redshift
mhalos = np.geomspace(1e10, 1e13, mass_bins)
mhalos = np.broadcast_to(mhalos, (len(zobs), mhalos.size)) * u.Msun
print(zobs)
print(mhalos[0])


# calculate corresponding virial temperatures
def virial_T(mhalo, Rvir):
    """Halo fp

    Args:
        mhalo (_type_): halo mass
        Rvir (_type_): halo virial radius

    Returns:
        _type_: virial temperture
    """
    # Ensure mhalo and Rvir have units
    if not isinstance(mhalo, u.Quantity):
        mhalo = mhalo * u.Msun
    if not isinstance(Rvir, u.Quantity):
        Rvir = Rvir * u.kpc

    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return ((2 / 5) * ((G * mhalo * mp) / (Rvir * kb))).to(u.K)


def virial_radius(z, mhalo, Delc=200):
    """
    Halo virial radius, classical 200 top-hat overdensity.
    Supports z as a 1D array and mhalo as a 2D array (shape: [len(z), N]).
    Args:
        z (array-like): Redshift array of shape (len(z),)
        mhalo (array-like): Halo mass array of shape (len(z), N)
        Delc (float): Overdensity parameter (default: 200)
    Returns:
        ndarray: Virial radius array of shape (len(z), N)
    """
    z = np.asarray(z)

    # Ensure mhalo has units
    if not isinstance(mhalo, u.Quantity):
        mhalo = mhalo * u.Msun

    # Broadcast critical density to shape (len(z), 1)
    rhoc = LCDM.critical_density(z)[:, np.newaxis]
    rvir = (mhalo / (rhoc * (4 / 3) * np.pi * Delc)) ** (1 / 3)

    return rvir.to(u.kpc)


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def vcirc_from_virial_T(Tvir, mu=0.59):
    kb = consts.k_B
    mp = consts.m_p
    return np.sqrt((2 * kb * Tvir) / (mu * mp)).to(u.km / u.s)


# %% run grid of models for comparison using baseline

# write to file
file = "./runs/smhm_baseline_redshift_scan_more_halo_obsfig2.h5"

if not os.path.exists(file):
    redshift_variation, zsims = run_baseline_model_redshift_grid(
        observe_at=zobs,  # redshift we want to observe
        mhalos=mhalos,
        write_to_file=file,
    )
else:
    print("file already exists")
    with h5py.File(file, "r") as f:
        smhm_normalized_C23 = f["SMHM"][
            :
        ]  # smhm is already normalized by baryon fractions
        mhalo_obs_C23 = f["Mhalo_obs"][:]
        zobs_C23 = f["redshifts"][:]
        print(f.keys())
        print(f["redshifts"][:])
models_label_C23 = "C23 model"

# %%
# write to file
# file = "./runs/smhm_2phase_redshift_scan_KS_kappa0p02_n1p8_r0p018.h5"
kappa_sfr = 0.02
n_sfr = 1.8
r_disk_sfr = 0.018
eta_z = 0.7
param_txt = (
    f"redshift_scan_KS_1998_16bins_Radau_{str(kappa_sfr).replace('.', 'p')}_"
    + f"n{str(n_sfr).replace('.', 'p')}_"
    + f"r{str(r_disk_sfr).replace('.', 'p')}"
    + f"_etaZ{str(eta_z).replace('.', 'p')}"
)
file = "./runs/smhm_2phase_redshift_scan_" f"{param_txt}.h5"


if not os.path.exists(file):
    print("running 2-phase model grid...", file)
    redshift_variation, zsims = run_2phase_model_redshift_grid(
        observe_at=zobs,  # redshift we want to observe
        mhalos=mhalos,
        write_to_file=file,
        disk_scale_length=r_disk_sfr,
        KS_n=n_sfr,
        KS_kappa_s=kappa_sfr,
        eta_z=eta_z,
    )
else:
    print("file already exists")
    with h5py.File(file, "r") as f:
        smhm_normalized = f["SMHM"][:]  # smhm is already normalized by baryon fractions
        mhalo_obs = f["Mhalo_obs"][:]
        zobs = f["redshifts"][:]
        print(f.keys())
        print(f["redshifts"][:])
models_label = ""

# Calculate for C23 baseline model
rvirs_C23 = virial_radius(zobs_C23, mhalo_obs_C23)  # virial radius in kpc
halo_Tvirs_C23 = virial_T(mhalo_obs_C23, rvirs_C23)  # virial temperature in K
vcirc_C23 = vcirc_from_virial_T(halo_Tvirs_C23)

# Calculate for 2-phase model
rvirs = virial_radius(zobs, mhalos)  # virial radius in kpc
halo_Tvirs = virial_T(mhalos, rvirs)  # virial temperature in K
vcirc = vcirc_from_virial_T(halo_Tvirs)

# %%

fig, ax = plt.subplots(2, 2, figsize=(10, 6.25), dpi=300, sharex="row", sharey="row")
plt.subplots_adjust(hspace=0.23, wspace=0.05)
ax = ax.ravel()

# Use the colors in the loop

ax[2].plot(
    [1e6, 1e6],
    [3e-3, 0.45],
    color="w",
    lw=2,
    ls=":",
    label=r"${\rm Shuntov\:et.\:al.\:24 \: observations}$",
)
ax[2].plot(
    [1e6, 1e6],
    [3e-3, 0.45],
    color="k",
    lw=2,
    ls=":",
    label=r"$T_{\rm vir} \geq 10^6$ K",
)

for i, zb in enumerate(zbins_str[::-1]):
    
    ax[1].plot(
        halo_Tvirs_C23[i],
        smhm_normalized_C23[i],
        color=cmap_colors[::-1][i],
        lw=2,
        label=f"${zbins_ctr[i]}$",
    )
    # Scatter for halo with Tvir < 1e6
    mask_scatter_C23 = halo_Tvirs_C23[i].value < 1e6
    ax[1].scatter(
        halo_Tvirs_C23[i][mask_scatter_C23],
        smhm_normalized_C23[i][mask_scatter_C23],
        color=cmap_colors[::-1][i],
        s=25,
        marker="o",
        edgecolor="k",
        linewidth=0.5,
        zorder=5,
    )
    # Scatter for halo with Tvir >= 1e6 (no edge color)
    # ax[1].scatter(
    #     halo_Tvirs_C23[i][~mask_scatter_C23],
    #     smhm_normalized_C23[i][~mask_scatter_C23],
    #     color=cmap_colors[::-1][i],
    #     s=25,
    #     marker="o",
    #     zorder=5,
    # )
    # now plot for the latest model
    ax[0].plot(
        halo_Tvirs[i],
        smhm_normalized[i],
        color=cmap_colors[::-1][i],
        lw=2,
        label=f"${zbins_ctr[i]}$",
    )
    
    # Scatter for halo with Tvir < 1e6
    mask_scatter = halo_Tvirs[i].value < 1e6
    ax[0].scatter(
        halo_Tvirs[i][mask_scatter],
        smhm_normalized[i][mask_scatter],
        color=cmap_colors[::-1][i],
        s=25,
        marker="o",
        edgecolor="k",
        linewidth=0.5,
        zorder=5,
    )
    # Scatter for halo with Tvir >= 1e6 (no edge color)
    # ax[0].scatter(
    #     halo_Tvirs[i][~mask_scatter],
    #     smhm_normalized[i][~mask_scatter],
    #     color=cmap_colors[::-1][i],
    #     s=25,
    #     marker="o",
    #     zorder=5,
    # )

    smhm_shuntov = np.array(Mstar[zb]) / (np.array(Mhalo[zb]) * (Ob0 / Omegam0))
    smhm_shuntov_low = np.array(Mstar_low[zb]) / (np.array(Mhalo[zb]) * (Ob0 / Omegam0))
    smhm_shuntov_up = np.array(Mstar_up[zb]) / (np.array(Mhalo[zb]) * (Ob0 / Omegam0))
    ax[3].fill_between(
        Mhalo[zb],
        smhm_shuntov_low,
        smhm_shuntov_up,
        lw=0,
        alpha=0.3,
        color=cmap_colors[::-1][i],
        label=f"${zb}$",
    )
    ax[2].fill_between(
        Mhalo[zb],
        smhm_shuntov_low,
        smhm_shuntov_up,
        lw=0,
        alpha=0.3,
        color=cmap_colors[::-1][i],
        label=f"${zb}$",
    )

    # Separate masks for C23 and latest model
    mask_C23 = halo_Tvirs_C23[i].value < 1e6
    mask_latest = halo_Tvirs[i].value < 1e6

    ax[3].plot(
        mhalo_obs_C23[i][mask_C23],
        smhm_normalized_C23[i][mask_C23],
        color=cmap_colors[::-1][i],
        lw=2,
    )
    ax[3].plot(
        mhalo_obs_C23[i][~mask_C23],
        smhm_normalized_C23[i][~mask_C23],
        color=cmap_colors[::-1][i],
        lw=2,
        alpha=0.8,
        ls=":",
    )

    # make a dotted line, straight between mask and ~mask to connect
    ax[3].plot(
        [mhalo_obs_C23[i][mask_C23][-1], mhalo_obs_C23[i][~mask_C23][0]],
        [smhm_normalized_C23[i][mask_C23][-1], smhm_normalized_C23[i][~mask_C23][0]],
        color=cmap_colors[::-1][i],
        lw=2,
        ls=":",
    )

    # now for the latest model
    ax[2].plot(
        mhalo_obs[i][mask_latest],
        smhm_normalized[i][mask_latest],
        color=cmap_colors[::-1][i],
        lw=2,
    )
    ax[2].plot(
        mhalo_obs[i][~mask_latest],
        smhm_normalized[i][~mask_latest],
        color=cmap_colors[::-1][i],
        lw=2,
        alpha=0.8,
        ls=":",
    )
    # make a dotted line, straight between mask and ~mask to connect
    ax[2].plot(
        [mhalo_obs[i][mask_latest][-1], mhalo_obs[i][~mask_latest][0]],
        [smhm_normalized[i][mask_latest][-1], smhm_normalized[i][~mask_latest][0]],
        color=cmap_colors[::-1][i],
        lw=2,
        ls=":",
    )


ax[3].set(
    xlabel=r"$M_{\rm halo} \: {\rm [M_\odot]}$",
   
    yscale="log",
    xscale="log",
    xlim=(1e10, 9e12),
    ylim=(3e-3, 0.8),
)
ax[1].set(
    xlabel=r"$T_{\rm vir} \: {\rm [K]}$",
    
    yscale="log",
    xscale="log",
    xlim=(3e4, 5e7),
    ylim=(3e-3, 0.8),
)
ax[0].set(
    xlabel=r"$T_{\rm vir} \: {\rm [K]}$",
    yscale="log",
    xscale="log",
    xlim=(3e4, 5e7),
     ylabel=r"$M_{\star}$/ $M_{\rm halo} (\Omega_{\rm b}/\Omega_{\rm m})$",
)
ax[2].set(
    xlabel=r"$M_{\rm halo} \: {\rm [M_\odot]}$",
    yscale="log",
    xscale="log",
    xlim=(1e10, 9e12),
    ylabel=r"$M_{\star}$/ $M_{\rm halo} (\Omega_{\rm b}/\Omega_{\rm m})$",
)

# Add a twin x-axis to ax[0] showing circular velocity corresponding to T_vir

# Get current x-ticks for T_vir axis
tvir_ticks = ax[1].get_xticks()
# Remove ticks outside the plotting range
tvir_ticks = tvir_ticks[
    (tvir_ticks >= ax[1].get_xlim()[0]) & (tvir_ticks <= ax[1].get_xlim()[1])
]

# Convert T_vir ticks to circular velocity
vcirc_ticks = vcirc_from_virial_T(tvir_ticks * u.K).value

# Create twin axis
ax1_vcirc = ax[1].twiny()
ax1_vcirc.set_xscale("log")
ax1_vcirc.set_xlim(ax[1].get_xlim())
ax1_vcirc.set_xticks(tvir_ticks)
ax1_vcirc.set_xticklabels(["{:.0f}".format(v) for v in vcirc_ticks])
ax1_vcirc.set_xlabel(r"$v_{\rm circ} \:  {\rm [km/s]}$ ", labelpad=10)
# do the next column too
ax0_vcirc = ax[0].twiny()
ax0_vcirc.set_xscale("log")
ax0_vcirc.set_xlim(ax[0].get_xlim())
ax0_vcirc.set_xticks(tvir_ticks)
ax0_vcirc.set_xticklabels(["{:.0f}".format(v) for v in vcirc_ticks])
ax0_vcirc.set_xlabel(r"$v_{\rm circ} \:  {\rm [km/s]}$ ", labelpad=10)

# ax0_vcirc.minorticks_off()

# add z = 0
ax[3].fill_between(
    10**loghm,
    smhm_behroozi * smhm_err_low,
    smhm_behroozi * smhm_err_up,
    facecolor="grey",
    alpha=0.3,
    zorder=0,
)
ax[2].fill_between(
    10**loghm,
    smhm_behroozi * smhm_err_low,
    smhm_behroozi * smhm_err_up,
    facecolor="grey",
    alpha=0.3,
    zorder=0,
)
ax[2].text(
    0.05,
    0.01,
    r"$z=0$ (Behroozi et al. 2019)",
    transform=ax[2].transAxes,
    fontsize=8,
    rotation=32,
    ha="left",
)

ax[1].plot(
    halo_Tvirs_C23[-1],
    smhm_normalized_C23[-1],
    color="grey",
    lw=2,
    label="0.0",
    alpha=0.8,
)
# Scatter for halo with Tvir < 1e6
mask_z0_C23 = halo_Tvirs_C23[-1].value < 1e6
ax[1].scatter(
    halo_Tvirs_C23[-1][mask_z0_C23],
    smhm_normalized_C23[-1][mask_z0_C23],
    color="grey",
    s=25,
    marker="o",
    edgecolor="k",
    linewidth=0.5,
    zorder=5,
    alpha=0.8,
)
# Scatter for halo with Tvir >= 1e6 (no edge color)
# ax[1].scatter(
#     halo_Tvirs_C23[-1][~mask_z0_C23],
#     smhm_normalized_C23[-1][~mask_z0_C23],
#     color="grey",
#     s=25,
#     marker="o",
#     zorder=5,
#     alpha=0.8,
# )
ax[0].plot(
    halo_Tvirs[-1],
    smhm_normalized[-1],
    color="grey",
    lw=2,
    label="0.0",
    alpha=0.8,
)

# Scatter for halo with Tvir < 1e6
mask_z0 = halo_Tvirs[-1].value < 1e6
ax[0].scatter(
    halo_Tvirs[-1][mask_z0],
    smhm_normalized[-1][mask_z0],
    color="grey",
    s=25,
    marker="o",
    edgecolor="k",
    linewidth=0.5,
    zorder=5,
    alpha=0.8,
)
# Scatter for halo with Tvir >= 1e6 (no edge color)
# ax[0].scatter(
#     halo_Tvirs[-1][~mask_z0],
#     smhm_normalized[-1][~mask_z0],
#     color="grey",
#     s=25,
#     marker="o",
#     zorder=5,
#     alpha=0.8,
# )

ax[1].text(
    0.05,
    0.95,
    models_label_C23,
    transform=ax[1].transAxes,
    ha="left",
    va="top",
    fontsize=10,
)


# make the z=0 line with mask
# Create mask for Tvir < 1e6
mask = halo_Tvirs[-1].value < 1e6
mask_C23 = halo_Tvirs_C23[-1].value < 1e6
ax[3].plot(
    mhalo_obs_C23[-1][mask_C23], smhm_normalized_C23[-1][mask_C23], color="grey", lw=2, alpha=1
)
ax[3].plot(
    mhalo_obs_C23[-1][~mask_C23],
    smhm_normalized_C23[-1][~mask_C23],
    color="grey",
    lw=2,
    alpha=0.8,
    ls=":",
)
ax[2].plot(mhalo_obs[-1][mask], smhm_normalized[-1][mask], color="grey", lw=2, alpha=1)
ax[2].plot(
    mhalo_obs[-1][~mask],
    smhm_normalized[-1][~mask],
    color="grey",
    lw=2,
    alpha=0.8,
    ls=":",
)

# make a dotted line, straight between mask and ~mask to connect
ax[3].plot(
    [mhalo_obs_C23[-1][mask_C23][-1], mhalo_obs_C23[-1][~mask_C23][0]],
    [smhm_normalized_C23[-1][mask_C23][-1], smhm_normalized_C23[-1][~mask_C23][0]],
    color="grey",
    lw=2,
    ls=":",
)
#  make a text annotation indicating that the dotted lines are Tvir >= 1e6 K
# plot a dotted line somewhere out the frame

ax[2].legend(
    frameon=False,
    fontsize=9,
    ncol=5,
    title_fontsize=10,
    bbox_to_anchor=(1, -0.2),
    loc="upper center",
)
ax[0].legend(
    frameon=False,
    fontsize=8,
    ncol=3,
    title=r"$z$",
    title_fontsize=10,
    loc="lower right",
)

# ax[0].text(
#     0.05,
#     0.95,
#     r"this work"
#     "\n"
#     r" $\kappa_s={}$"
#     "\n"
#     r" $n={}$"
#     "\n"
#     r" $r_{{\rm disk}}={}$".format(kappa_sfr, n_sfr, r_disk_sfr),
#     transform=ax[0].transAxes,
#     ha="left",
#     va="top",
# )
ax[0].text(
    0.05,
    0.95,
    r"this work",
    transform=ax[0].transAxes,
    ha="left",
    va="top",
    fontsize=10
)
plt.savefig(
    "./final_figs/fig_8_comparison_SMHM_KS_kap0p02_rd_0p018_n_1p8_etaZ_0p7.pdf",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

# %%
