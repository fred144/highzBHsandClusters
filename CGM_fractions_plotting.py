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

# from cgm_sf_regulator import CGMRegulator, mhalo_at_z0, halo_diagnostics_v2
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


# calculate corresponding virial temperatures
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


zbins_str = [
    "0.2 < z < 0.5",
    "0.5 < z < 0.8",
    # "0.8 < z < 1.1",
    "1.1 < z < 1.5",
    # "1.5 < z < 2.0",
    "2.0 < z < 2.5",
    # "2.5 < z < 3.0",
    "3.0 < z < 3.5",
    # "3.5 < z < 4.5",
    "4.5 < z < 5.5",
    # "5.5 < z < 6.5",
    "6.5 < z < 7.5",
    # "7.5 < z < 8.5",
    "8.5 < z < 10.0",
    # "10.0 < z < 12.0",
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

mhalos = np.geomspace(1e10, 1e13, 8)  # make a unique halo array for each redshift
mhalos = np.broadcast_to(mhalos, (len(zbins_ctr), mhalos.size)) * u.Msun
print(zbins_ctr)
print(mhalos[0])

# add grey to the cmap at the end
cmap_colors.append((0.5, 0.5, 0.5))  # add grey color

# %%
# write to file
file = "./runs/smhm_2phase_redshift_scan_redshift_scan_KS_1998_16bins_Radau_0p02_n1p8_r0p018_etaZ0p7.h5"

if not os.path.exists(file):
    redshift_variation, zsims = run_2phase_model_redshift_grid(
        observe_at=zbins_ctr,  # redshift we want to observe
        mhalos=mhalos,
        write_to_file=file,
    )
else:
    print("file already exists")
    f = h5py.File(file, "r")
    smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
    mhalo_obs = f["Mhalo_obs"]
    print(f.keys())
    print(f["redshifts"][:])
# %%

fig, ax = plt.subplots(
    2,
    2,
    figsize=(10.25, 5.5),
    dpi=300,
    gridspec_kw={"height_ratios": [3, 1.5]},
    sharex=True,
)
plt.subplots_adjust(hspace=0.05)

f = h5py.File(file, "r")
mhalo_obs = f["Mhalo_obs"][:]  # shape: (n_z, n_m)
mcgm_cold = f["MCGM_cold_obs"][:]  # shape: (n_z, n_m)
mcgm_hot = f["MCGM_hot_obs"][:]  # shape: (n_z, n_m)
redshifts = f["redshifts"][:]  # shape: (n_z,)
mcgm_total = mcgm_cold + mcgm_hot
rvirs = virial_radius(redshifts, mhalo_obs * u.Msun)  # virial radius in kpc
halo_Tvirs = virial_T(mhalo_obs * u.Msun, rvirs)  # virial temperature in K
for i, z in enumerate(redshifts):
    mask = halo_Tvirs[i] < 1e6 * u.K

    ax[0, 0].plot(mhalo_obs[i][mask], mcgm_cold[i][mask], color=cmap_colors[i], lw=2)
    ax[0, 1].plot(
        mhalo_obs[i][mask],
        mcgm_hot[i][mask],
        color=cmap_colors[i],
       
        lw=2,
    )
    ax[1, 0].plot(
        mhalo_obs[i][mask],
        mcgm_cold[i][mask] / mcgm_total[i][mask],
        color=cmap_colors[i],
       
        lw=2,
    )
    ax[1, 1].plot(
        mhalo_obs[i][mask],
        mcgm_hot[i][mask] / mcgm_total[i][mask],
        color=cmap_colors[i],
        label=f"${z:.1f}$",
        lw=2,
    )

    # Plot the "rest" (where Tvir >= 1e6 K) as dotted lines
    # Find the transition index where Tvir crosses 1e6 K
    rest_mask = ~mask
    if np.any(rest_mask) and np.any(mask):
        # find the first index where rest_mask is True
        first_rest_idx = np.argmax(rest_mask)
        # include the last valid (mask=True) value before the transition
        last_valid_idx = first_rest_idx - 1 if first_rest_idx > 0 else 0

        # prepare arrays including the overlap point
        mhalo_overlap = np.concatenate(
            ([mhalo_obs[i][last_valid_idx]], mhalo_obs[i][rest_mask])
        )
        mcgm_cold_overlap = np.concatenate(
            ([mcgm_cold[i][last_valid_idx]], mcgm_cold[i][rest_mask])
        )
        mcgm_hot_overlap = np.concatenate(
            ([mcgm_hot[i][last_valid_idx]], mcgm_hot[i][rest_mask])
        )
        mcgm_total_overlap = np.concatenate(
            ([mcgm_total[i][last_valid_idx]], mcgm_total[i][rest_mask])
        )

        ax[0, 0].plot(
            mhalo_overlap, mcgm_cold_overlap, color=cmap_colors[i], lw=2, ls=":"
        )
        ax[0, 1].plot(
            mhalo_overlap, mcgm_hot_overlap, color=cmap_colors[i], lw=2, ls=":"
        )
        ax[1, 0].plot(
            mhalo_overlap,
            mcgm_cold_overlap / mcgm_total_overlap,
            color=cmap_colors[i],
            lw=2,
            ls=":",
        )
        ax[1, 1].plot(
            mhalo_overlap,
            mcgm_hot_overlap / mcgm_total_overlap,
            color=cmap_colors[i],
            lw=2,
            ls=":",
        )


ax[0, 0].set(
    xscale="log",
    yscale="log",
    # xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,cold}}$ [M$_{\odot}$]",
    xlim=(1e10, 9.5e12),
    ylim=(2e5, 2e11),
)
ax[0, 1].set(
    xscale="log",
    yscale="log",
    # xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,hot}}$ [M$_{\odot}$]",
    xlim=(1e10, 9.5e12),
    ylim=(2e5, 2e11),
)
ax[1, 0].set(
    xscale="log",
    xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    # ylabel=r"$M_{\mathrm{CGM,cold}} / M_{\mathrm{CGM}}$",
    ylabel=r"${\rm CGM~ mass~ fraction }$",
    xlim=(1e10, 9.5e12),
    ylim=(-0.1, 0.75),
)
ax[1, 1].set(
    xscale="log",
    xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    xlim=(1e10, 9.5e12),
    ylim=(0.20, 1.1),
)
# ax[1, 1].set(
#     xscale="log",
#     xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
#     ylabel=r"$M_{\mathrm{CGM,hot}} / M_{\mathrm{CGM}}$",
#     xlim=(1e10, 9.5e12),
#     ylim=(-0.1, 1.1),
# )


# add baryonic limit as shaded region
baryon_limit = f_b * mhalo_obs[-1]
ax[0, 0].fill_between(mhalo_obs[-1], baryon_limit, 1e12, alpha=0.5, facecolor="grey")
ax[0, 0].text(
    2e10,
    5e9,
    r"$f_{\rm b} M_{\rm halo}$",
    fontsize=10,
    rotation=18,
    color="black",
    ha="left",
)
ax[0, 1].fill_between(mhalo_obs[-1], baryon_limit, 1e12, alpha=0.5, facecolor="grey")


# ================= OBSERVATIONS
## fearman and werk 23, https://iopscience.iop.org/article/10.3847/1538-4357/acf217/pdf
##0.1  <z < 0.4 galaxies observed in the COS-Halos/eCGM surveys.
mh_fw23 = 1e12
mcool_fw23 = 3e9
mcool_fw23_dex_scatter = 0.5
mcool_fw23_err_lower = (
    10 ** (np.log10(mcool_fw23) - mcool_fw23_dex_scatter) - mcool_fw23
)
mcool_fw23_err_upper = (
    10 ** (np.log10(mcool_fw23) + mcool_fw23_dex_scatter) - mcool_fw23
)
mcool_fw23_err = [[-mcool_fw23_err_lower], [mcool_fw23_err_upper]]
ax[0, 0].errorbar(
    mh_fw23,
    mcool_fw23,
    yerr=mcool_fw23_err,
    fmt="o",
    color="black",
    label=r"COS-Halos, $z = 0.1 - 0.4$ (Fearman \& Werk 2023)",
    # label=r"COS-Halos (Faerman \& Werk 2023)",
    zorder=10,
    markersize=6,
    lw=1.5,
    alpha=0.7,
)

# Yong Zheng et al. 2024 https://iopscience.iop.org/article/10.3847/1538-4357/acfe6b/pdf
mh_range_zheng24 = (10**10, 10**11.5)
mh_median_zheng24 = 10**10.9
mcool_zheng24 = 10**8.4
ax[0, 0].errorbar(
    mh_median_zheng24,
    mcool_zheng24,
    xerr=[
        [mh_median_zheng24 - mh_range_zheng24[0]],
        [mh_range_zheng24[1] - mh_median_zheng24],
    ],
    fmt="s",
    color="k",
    label=r"nearby dwarfs, $z = 0.003 - 0.3$ (Zheng et al. 2024)",
    zorder=10,
    markersize=6,
    lw=1.5,
    alpha=0.7,
    
)

# upper limits on dwarfs faerman et al https://ui.adsabs.harvard.edu/abs/2025ApJ...982L..30F/abstract
mh_bin_1 = (10**10.2, 10**10.8)
mh_bin_2 = (10**10.8, 10**11.15)
mh_bin_3 = (10**11.15, 10**11.5)

mh_centers = [3.2e10, 7.9e10, 2e11]  # actual centers of the bins

mh_errs = [
    [mh_centers[i] - mh_bin[0] for i, mh_bin in enumerate([mh_bin_1, mh_bin_2, mh_bin_3])],
    [mh_bin[1] - mh_centers[i] for i, mh_bin in enumerate([mh_bin_1, mh_bin_2, mh_bin_3])],
]

mcold_clumpy = [4.2e7, 9.5e7, 1.4e8]
ax[0, 0].errorbar(
    mh_centers,
    mcold_clumpy,
    xerr=mh_errs,
    fmt="^",
    color="k",
    label=r"nearby dwarfs, clumpy CGM (Faerman et al. 2025)",
    zorder=10,
    markersize=6,
    lw=1.5,
    alpha=0.7,
)

# COS-Halos, Werk+2014 https://arxiv.org/abs/1403.0947
# mhalo_werk24 = 10**12.2 # msun
# mcool_werk24_lower = 7e10
# mcool_werk24_upper = 12e10
# mcool_werk24_cold_mass = (mcool_werk24_lower + mcool_werk24_upper) / 2
# ax[0, 0].errorbar(
#     mhalo_werk24,
#     mcool_werk24_cold_mass,
#     yerr=[[mcool_werk24_cold_mass - mcool_werk24_lower], [mcool_werk24_upper - mcool_werk24_cold_mass]],
#     fmt="o",
#     color="tab:green",
#     label="COS-Halos (Werk+2014)",
#     zorder=10,
#     markersize=6,
#     lw=1.5,
# )

## zahedy +19 (https://academic.oup.com/mnras/article/484/2/2257/5256659#:~:text=halo%20mass%20in%20COS%2DLRG)
# z ~0.4
mh_zahedy19 = 10**13
mcool_z19_low = 1e10
mcool_z19_up = 2e10
mcool_z19 = (mcool_z19_low + mcool_z19_up) / 2
# ax[0, 0].errorbar(
#     mh_zahedy19,
#     mcool_z19,
#     yerr=[[mcool_z19 - mcool_z19_low], [mcool_z19_up - mcool_z19]],
#     fmt="s",
#     color="black",
#     # label="Zahedy et al. 2019",
#     zorder=10,
#     markersize=5,
#     lw=2,
# )

## werk et al 2014, figure 11 https://iopscience.iop.org/article/10.1088/0004-637X/792/1/8/pdf
mh_werk14 = 10**12.2
mhot_werk14_range = [1e9, 14e9]
mhot_werk14 = (mhot_werk14_range[0] + mhot_werk14_range[1]) / 2
mhot_werk_err = [[mhot_werk14 - mhot_werk14_range[0]], [mhot_werk14_range[1] - mhot_werk14]]

mwarm_werk14_range = [1e10, 1e11]
mwarm_werk14 = (mwarm_werk14_range[0] + mwarm_werk14_range[1]) / 2
mwarm_werk_err = [[mwarm_werk14 - mwarm_werk14_range[0]], [mwarm_werk14_range[1] - mwarm_werk14]]

mwarmhot_werk14_total = mwarm_werk14 + mhot_werk14

err_lower = np.sqrt(mwarm_werk_err[0][0]**2 + mhot_werk_err[0][0]**2)
err_upper = np.sqrt(mwarm_werk_err[1][0]**2 + mhot_werk_err[1][0]**2)
mwarmhot_werk14_total_err = [[err_lower], [err_upper]]
ax[0, 1].errorbar(
    mh_werk14,
    mwarmhot_werk14_total,
    yerr=mwarmhot_werk14_total_err,
    fmt="D",
    color="k",
    # label=r"COS-Halos, warm+hot" "\n""(Tumlinson et al. 2011; Peeples et al. 2014;" "\n" " Anderson et al. 2013; Werk et al. 2014)",
    label=r"COS-Halos, warm + hot",
    zorder=10,
    markersize=6,
    lw=1.5,
    alpha=0.7,
)


### marvels
import pandas as pd

df = pd.read_csv("./data/marvel_dwarfs.csv")
marvels_m200c = 10**df["M_200c"].values   # convert to Msun
marvels_mcgmwarm = 10**df["M_Warm_CGM"].values
# ax[0, 1].scatter(
#     marvels_m200c,
#     marvels_mcgmwarm,
#     facecolor="k",
#     marker="v",
#     label=r"Marvel-ous Dwarfs warm CGM (Piacitelli et al. 2024)",
#     zorder=0,
#     s=10,
#     alpha=0.5,
# )

# ====================end observations


ax[1, 1].legend(
    frameon=False,
    loc="lower center",
    fontsize=11,
    ncol=9,
    title=r"$z$",
    bbox_to_anchor=(-0.1, 3.1),
    handletextpad=0.3,
)
ax[0, 1].legend(frameon=False, loc="lower right", fontsize=10, ncol=1, handletextpad=0.1, columnspacing=0.5)
# rescale legend markers
ax[0, 0].legend(frameon=False, loc="lower right", fontsize=8,  ncols=1, handletextpad=0.1, columnspacing=0.5, markerscale=0.8)

# show a grid
# ax[0, 0].grid(alpha=0.3, which="both")
# ax[0, 1].grid(alpha=0.3, which="both")
# ax[1, 0].grid(alpha=0.3, which="both")
# ax[1, 1].grid(alpha=0.3, which="both")


for axes in ax.ravel():
    for line in axes.lines:
        line.set_zorder(1)

plt.savefig(
    "./final_figs/fig_9_twophase_CGM_fractions.pdf",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

# %%
