# %% need to see what the results of varying the normalization kappa in the KS law
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmasher as cmr
import astropy.units as u
from astropy import constants as consts
from astropy import cosmology
from run_grids_of_models import (
    run_baseline_model_redshift_grid,
    run_2phase_model_redshift_grid,
)
import os
from astropy.table import Table

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        # "text.usetex": True,
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
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)
# %%
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

# %%
mass_bins = 10
zobs = zbins_ctr
# make a unique halo array for each redshift
mhalos = np.geomspace(1e10, 1e13, mass_bins)
mhalos = np.broadcast_to(mhalos, (len(zobs), mhalos.size)) * u.Msun
print(zobs)
print(mhalos[0])

# kappa_sfr = 0.1
# kapppa_sfrs = np.geomspace(0.01, 10, 20)
kappa_sfr = 0.1
# n_sfrs = np.linspace(1.3, 1.5, 10)
# r_disk_sfrs = np.geomspace(0.011, 0.1, 15)
r_disk_sfr = 0.02
# n_sfr = 1.5
n_sfrs = np.linspace(1.0, 2.0, 20)
for j, n_sfr in enumerate(n_sfrs, start=15):

    param_txt = (
        f"KSv2_kappa{str(kappa_sfr).replace('.', 'p')}_"
        + f"n{str(n_sfr).replace('.', 'p')}_"
        + f"r{str(r_disk_sfr).replace('.', 'p')}"
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
        )

    f = h5py.File(file, "r")
    smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
    mhalo_obs = f["Mhalo_obs"]
    mstar = f["Mstar_obs"]
    mism = f["MISM_obs"]
    zs = f["redshifts"][:]
    sfr = f["SFR_obs"]
    mmetal_cgm = f["MMetals_obs"]
    rvirs = virial_radius(zs, mhalo_obs * u.Msun)  # virial radius in kpc
    halo_Tvirs = virial_T(mhalo_obs * u.Msun, rvirs)  # virial temperature in K
    print(f.keys())
    print(f["redshifts"][:])

    # # write to file
    # file_baseline = "./runs/smhm_baseline_redshift_latest_more_info.h5"
    # mhalos = np.broadcast_to(mhalo_obs[0], (len(zs), mhalo_obs[0].size)) * u.Msun
    # # redshift_variation, zsims = run_baseline_model_redshift_grid(
    # #     observe_at=zs,  # redshift we want to observe
    # #     mhalos=mhalos,
    # #     write_to_file=file,
    # # )
    # f_baseline = h5py.File(file_baseline, "r")
    # smhm_baseline = f_baseline["SMHM"]  # smhm is already normalized by baryon fractions
    # mhalo_obs_baseline = f_baseline["Mhalo_obs"]
    # mstar_baseline = f_baseline["Mstar_obs"]
    # mism_baseline = f_baseline["MISM_obs"]
    # sfr_baseline = f_baseline["SFR_obs"]

    fig, ax = plt.subplots(2, 1, figsize=(5, 6.5), dpi=300, sharex=True, sharey="row")
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.05)
    # get a colormap from cmasher
    cmap = plt.get_cmap("Set1")
    colors = cmap(np.linspace(0, 1, len(zs)))
    Tvir_max = 1e6 * u.K
    # loop through redshift and plot mgas/mstar vs mstar
    for i, z in enumerate(zs):
        Tvirs = halo_Tvirs[i]
        mask = Tvirs < Tvir_max

        ax[0].plot(
            mstar[i],
            mism[i] / mstar[i],
            color=colors[i],
        )
        # ax[1].plot(
        #     mstar_baseline[i],
        #     mism_baseline[i] / mstar_baseline[i],
        #     color=colors[i],
        # )

        ax[0].scatter(
            mstar[i][mask],
            mism[i][mask] / mstar[i][mask],
            s=30,
            color=colors[i],
            edgecolor="k",
            zorder=3,
            label=f"{z:.1f}",
        )
        # ax[1].scatter(
        #     mstar_baseline[i][mask],
        #     mism_baseline[i][mask] / mstar_baseline[i][mask],
        #     s=30,
        #     color=colors[i],
        #     edgecolor="k",
        #     zorder=3,
        # )

        t_depletion = mism[i] / sfr[i]
        sSFR_Gyr = sfr[i] / mstar[i]  # Gyr^-1
        sSFR_yr = sSFR_Gyr * 1e-9
        ax[1].plot(
            mstar[i],
            sSFR_yr,
            color=colors[i],
        )
        ax[1].scatter(
            mstar[i][mask],
            sSFR_yr[mask],
            s=30,
            color=colors[i],
            edgecolor="k",
            zorder=3,
        )

        # t_depletion_baseline = mism_baseline[i] / sfr_baseline[i]
        # sSFR_Gyr_baseline = sfr_baseline[i] / mstar_baseline[i]  # Gyr^-1
        # sSFR_yr_baseline = sSFR_Gyr_baseline * 1e-9
        # ax[3].plot(
        #     mstar_baseline[i],
        #     sSFR_yr_baseline,
        #     color=colors[i],
        # )
        # ax[3].scatter(
        #     mstar_baseline[i][mask],
        #     sSFR_yr_baseline[mask],
        #     s=30,
        #     color=colors[i],
        #     edgecolor="k",
        #     zorder=3,
        # )

    # make a line passing through (8.7, 0.322and (10.35, -0.5) and extend it to the left and right

    # Calculate the slope and intercept in log-log space
    x1, y1 = 8.7, 0.2
    x2, y2 = 10.35, -0.4

    # Generate x values spanning the plot range
    x_vals = np.linspace(6, 12, 100)
    # Compute corresponding y values for the line in log-log space
    slope = (y2 - y1) / (x2 - x1)
    y_vals = y1 + slope * (x_vals - x1)

    ax[0].plot(
        10**x_vals,
        10**y_vals,
        ls="-",
        color="k",
        lw="4",
        alpha=0.5,
        zorder=1,
    )
    # put a text label near this line
    ax[0].text(
        0.85,
        0.22,
        r"Calette+18, $z\sim 0$",
        transform=ax[0].transAxes,
        fontsize=8,
        rotation=-35,
        va="center",
        ha="right",
    )

    ax[0].legend(
        ncols=4,
        frameon=False,
        bbox_to_anchor=(-0.05, 1.011),
        loc="lower left",
        title=r"redshift $z$",
        fontsize=10,
    )

    ax[0].set(xscale="log", yscale="log", ylabel=r"$M_{\rm {ISM}}/M_\star$")
    ax[1].set(
        xscale="log",
        yscale="log",
        xlabel=r"$M_\star$ [M$_\odot$]",
        ylabel=r"sSFR [yr$^{-1}$]",
        # xlim=(8e6, 8e11),
    )

    # add text for the KS parameters used
    ax[0].text(
        0.25,
        1.05,
        rf"KS: $\kappa={kappa_sfr:.3f}$, $n={n_sfr:.3f}$, $r_{{\rm disk}}={r_disk_sfr:.3f} R_{{\rm vir}}$",
        transform=ax[0].transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )

    plt.savefig(
        f"./figures/KS_ism_gas_fraction_n_scan_v2/{j:02d}_ism_gas_fractions_{param_txt}.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
# %% now, do single runs

mass_bins = 10
zobs = zbins_ctr
# make a unique halo array for each redshift
mhalos = np.geomspace(1e10, 1e13, mass_bins)
mhalos = np.broadcast_to(mhalos, (len(zobs), mhalos.size)) * u.Msun
print(zobs)
print(mhalos[0])
####
n_sfr = 1.5
r_disk_sfr = 0.02
kappa_sfr = 0.1
####
param_txt = (
    f"KS_kappa{str(kappa_sfr).replace('.', 'p')}_"
    + f"n{str(n_sfr).replace('.', 'p')}_"
    + f"r{str(r_disk_sfr).replace('.', 'p')}"
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
    )


f = h5py.File(file, "r")
smhm_normalized = f["SMHM"]  # smhm is already normalized by baryon fractions
mhalo_obs = f["Mhalo_obs"]
mstar = f["Mstar_obs"]
mism = f["MISM_obs"]
zs = f["redshifts"][:]
sfr = f["SFR_obs"]
mmetal_cgm = f["MMetals_obs"]
rvirs = virial_radius(zs, mhalo_obs * u.Msun)  # virial radius in kpc
halo_Tvirs = virial_T(mhalo_obs * u.Msun, rvirs)  # virial temperature in K
print(f.keys())
print(f["redshifts"][:])


fig, ax = plt.subplots(2, 1, figsize=(5, 6.5), dpi=300, sharex=True, sharey="row")
ax = ax.flatten()
plt.subplots_adjust(hspace=0.05)
# get a colormap from cmasher
cmap = plt.get_cmap("Set1")
colors = cmap(np.linspace(0, 1, len(zs)))
Tvir_max = 1e6 * u.K
# loop through redshift and plot mgas/mstar vs mstar
for i, z in enumerate(zs):
    Tvirs = halo_Tvirs[i]
    mask = Tvirs < Tvir_max

    ax[0].plot(
        mstar[i],
        mism[i] / mstar[i],
        color=colors[i],
    )

    ax[0].scatter(
        mstar[i][mask],
        mism[i][mask] / mstar[i][mask],
        s=30,
        color=colors[i],
        edgecolor="k",
        zorder=3,
        label=f"{z:.1f}",
    )

    t_depletion = mism[i] / sfr[i]
    sSFR_Gyr = sfr[i] / mstar[i]  # Gyr^-1
    sSFR_yr = sSFR_Gyr * 1e-9
    ax[1].plot(
        mstar[i],
        sSFR_yr,
        color=colors[i],
    )
    ax[1].scatter(
        mstar[i][mask], sSFR_yr[mask], s=30, color=colors[i], edgecolor="k", zorder=3
    )

# make a line passing through (8.7, 0.322and (10.35, -0.5) and extend it to the left and right

# Calculate the slope and intercept in log-log space
x1, y1 = 8.7, 0.2
x2, y2 = 10.35, -0.4

# Generate x values spanning the plot range
x_vals = np.linspace(6, 12, 100)
# Compute corresponding y values for the line in log-log space
slope = (y2 - y1) / (x2 - x1)
y_vals = y1 + slope * (x_vals - x1)

ax[0].plot(
    10**x_vals,
    10**y_vals,
    ls="-",
    color="k",
    lw="4",
    alpha=0.5,
    zorder=1,
)
# put a text label near this line
ax[0].text(
    0.85,
    0.22,
    r"Calette+18, $z\sim 0$",
    transform=ax[0].transAxes,
    fontsize=8,
    rotation=-35,
    va="center",
    ha="right",
)

ax[0].legend(
    ncols=4,
    frameon=False,
    bbox_to_anchor=(-0.05, 1.011),
    loc="lower left",
    title=r"redshift $z$",
    fontsize=10,
)

ax[0].set(xscale="log", yscale="log", ylabel=r"$M_{\rm {ISM}}/M_\star$")
ax[1].set(
    xscale="log",
    yscale="log",
    xlabel=r"$M_\star$ [M$_\odot$]",
    ylabel=r"sSFR [yr$^{-1}$]",
    # xlim=(8e6, 8e11),
)


# add text for the KS parameters used
ax[0].text(
    0.25,
    1.05,
    rf"KS: $\kappa={kappa_sfr:.4f}$, $n={n_sfr:.4f}$, $r_{{\rm disk}}={r_disk_sfr:.4f} R_{{\rm vir}}$",
    transform=ax[0].transAxes,
    fontsize=10,
    va="bottom",
    ha="left",
)

# plt.savefig(
#     f"./figures/KS_ism_gas_fraction_rdisk_scan/{j:02d}_ism_gas_fractions_{param_txt}.png",
#     dpi=200,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
plt.show()
# %% metallicity plot

z_sun = 0.0134
fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), dpi=300)
for i, z in enumerate(zs):
    metallicity = (mmetal_cgm[i] / mism[i]) / z_sun
    ax2.plot(
        mhalo_obs[i],
        metallicity,
        label=f"{z:.1f}",
        marker="o",
        color=colors[i],
        markeredgecolor="k",
    )

ax2.set(
    xscale="log",
    yscale="log",
    xlabel=r"$M_\star$ [M$_\odot$]",
    ylabel=r"CGM Metallicity [$Z_\odot$]",
)
ax2.legend(
    ncols=4,
    frameon=False,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    title=r"redshift $z$",
    fontsize=10,
)
plt.savefig(
    "./figures/cgm_metallicity_2phase.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
