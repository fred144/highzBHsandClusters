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
from cgm_sf_regulator import CGMRegulator, mhalo_at_z0, halo_diagnostics_v2
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
file = "./runs/smhm_2phase_redshift_scan_fig5.h5"

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
    2, 2, figsize=(10, 6), dpi=300, gridspec_kw={"height_ratios": [2, 1]}, sharex=True
)
plt.subplots_adjust(hspace=0.05)

f = h5py.File(file, "r")
mhalo_obs = f["Mhalo_obs"][:]  # shape: (n_z, n_m)
mcgm_cold = f["MCGM_cold_obs"][:]  # shape: (n_z, n_m)
mcgm_hot = f["MCGM_hot_obs"][:]  # shape: (n_z, n_m)
redshifts = f["redshifts"][:]  # shape: (n_z,)
mcgm_total = mcgm_cold + mcgm_hot

for i, z in enumerate(redshifts):
    ax[0, 0].plot(
        mhalo_obs[i], mcgm_cold[i], color=cmap_colors[i], label=f"$z={z:.1f}$"
    )
    ax[0, 1].plot(mhalo_obs[i], mcgm_hot[i], color=cmap_colors[i], label=f"$z={z:.1f}$")
    ax[1, 0].plot(
        mhalo_obs[i],
        mcgm_cold[i] / mcgm_total[i],
        color=cmap_colors[i],
        label=f"$z={z:.1f}$",
    )
    ax[1, 1].plot(
        mhalo_obs[i],
        mcgm_hot[i] / mcgm_total[i],
        color=cmap_colors[i],
        label=f"$z={z:.1f}$",
    )

ax[0, 0].set(
    xscale="log",
    yscale="log",
    # xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,cold}}$ [M$_{\odot}$]",
    xlim=(8e9, 2e13),
    ylim=(2e5, 2e12),
)
ax[0, 1].set(
    xscale="log",
    yscale="log",
    # xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,hot}}$ [M$_{\odot}$]",
    xlim=(8e9, 2e13),
    ylim=(2e5, 2e12),
)
ax[1, 0].set(
    xscale="log",
    xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,cold}} / M_{\mathrm{CGM,total}}$",
    xlim=(8e9, 2e13),
    ylim=(0, 1.1),
)
ax[1, 1].set(
    xscale="log",
    xlabel=r"$M_{\mathrm{halo}}$ [M$_{\odot}$]",
    ylabel=r"$M_{\mathrm{CGM,hot}} / M_{\mathrm{CGM,total}}$",
    xlim=(8e9, 2e13),
    ylim=(0, 1.1),
)

ax[0, 0].legend(
    frameon=False, loc="lower left", ncols=3, bbox_to_anchor=(0.5, 1.01), fontsize=11
)
plt.savefig(
    "./figures/fig5_2phase_CGMfracs.png", dpi=200, bbox_inches="tight", pad_inches=0.05
)
plt.show()
