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
file = "./runs/smhm_2phase_redshift_scan_KS_kap0p1_rd_0p02.h5"

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
    2, 2, figsize=(10, 5), dpi=300, gridspec_kw={"height_ratios": [3, 2]}, sharex=True,
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

   
    mask = halo_Tvirs[i] < 1e6 *u.K
    
    ax[0, 0].plot(
        mhalo_obs[i][mask], mcgm_cold[i][mask], color=cmap_colors[i], label=f"${z:.1f}$", lw=2
    )
    ax[0, 1].plot(mhalo_obs[i][mask], mcgm_hot[i][mask], color=cmap_colors[i], label=f"${z:.1f}$", lw=2)
    ax[1, 0].plot(
        mhalo_obs[i][mask],
        mcgm_cold[i][mask] / mcgm_total[i][mask],
        color=cmap_colors[i],
        label=f"${z:.1f}$",
        lw=2
    )
    ax[1, 1].plot(
        mhalo_obs[i][mask],
        mcgm_hot[i][mask] / mcgm_total[i][mask],
        color=cmap_colors[i],
        label=f"${z:.1f}$",
        lw=2
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
        mhalo_overlap = np.concatenate(([mhalo_obs[i][last_valid_idx]], mhalo_obs[i][rest_mask]))
        mcgm_cold_overlap = np.concatenate(([mcgm_cold[i][last_valid_idx]], mcgm_cold[i][rest_mask]))
        mcgm_hot_overlap = np.concatenate(([mcgm_hot[i][last_valid_idx]], mcgm_hot[i][rest_mask]))
        mcgm_total_overlap = np.concatenate(([mcgm_total[i][last_valid_idx]], mcgm_total[i][rest_mask]))

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
            ls=":"
        )
        ax[1, 1].plot(
            mhalo_overlap,
            mcgm_hot_overlap / mcgm_total_overlap,
            color=cmap_colors[i],
            lw=2,
            ls=":"
        )
    
    

ax[0, 0].set(
    xscale="log",
    yscale="log",
    # xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,cold}}$ [M$_{\odot}$]",
    xlim=(1e10, 1e13),
    ylim=(2e5, 3e11),
)
ax[0, 1].set(
    xscale="log",
    yscale="log",
    # xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,hot}}$ [M$_{\odot}$]",
    xlim=(1e10, 1e13),
    ylim=(1e7, 1e12),
)
ax[1, 0].set(
    xscale="log",
    xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,cold}} / M_{\mathrm{CGM}}$",
    xlim=(1e10, 1e13),
    ylim=(-0.1, 1.1),
)
ax[1, 1].set(
    xscale="log",
    xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,hot}} / M_{\mathrm{CGM}}$",
    xlim=(1e10, 1e13),
    ylim=(-0.1, 1.1),
)
ax[0, 0].legend(
    frameon=False, loc="lower right",  fontsize=9, ncol=2, title=r"redshift $z$"
)

for axes in ax.ravel():
    for line in axes.lines:
        line.set_zorder(1)
        
plt.savefig(
    "./figures/twophase_CGM_fractions.png", dpi=200, bbox_inches="tight", pad_inches=0.05
)
plt.show()

# %%
