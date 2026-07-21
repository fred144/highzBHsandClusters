# %%
# Quick look at the MULTIPLE_RUNS population grid (run_grids_of_models.py),
# reproducing only the "this work" column of smhm_vcirc_over_redshift_plotting.py
# (its ax[0]/ax[2]: SMHM vs T_vir and SMHM vs M_halo, colored by redshift) --
# no C23 baseline column. Each grid point is now a population of
# N_HALOS_TO_RUN realizations (one per t_init in START_TIMES), summarized
# as a median line + 16th/84th percentile band rather than a single value.
# The M_halo panel also carries the Shuntov et al. 2024 and Behroozi et al.
# 2019 (z=0) observational bands, same as the original script's bottom row.
import os
import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import astropy.constants as consts
import astropy.units as u
from astropy import cosmology
from astropy.table import Table

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
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

H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)

# same zbins_str / zbins_ctr as smhm_vcirc_over_redshift_plotting.py
zbins_str = [
    "0.2 < z < 0.5",
    "2.0 < z < 2.5",
    "2.5 < z < 3.0",
    "3.5 < z < 4.5",
    "5.5 < z < 6.5",
    "6.5 < z < 7.5",
    "7.5 < z < 8.5",
    "10.0 < z < 12.0",
]
zbins_ctr = []
for zb in zbins_str:
    z = zb.split("<")
    z = (float(z[0]) + float(z[2])) / 2
    zbins_ctr.append(z)
zbins_ctr = zbins_ctr[::-1]
zbins_ctr.append(0.01)

cmap = cmr.tropical
cmap_colors = [cmap(i / len(zbins_str)) for i in range(len(zbins_str))]


def virial_radius(z, mhalo, Delc=200):
    """
    Halo virial radius, classical 200 top-hat overdensity.
    Same as smhm_vcirc_over_redshift_plotting.py, but generalized to accept
    an extra trailing realization axis on mhalo (shape [len(z), N, n_runs])
    in addition to the plain [len(z), N] case.
    """
    z = np.asarray(z)
    if not isinstance(mhalo, u.Quantity):
        mhalo = mhalo * u.Msun
    rhoc = LCDM.critical_density(z)
    rhoc = rhoc.reshape((len(z),) + (1,) * (mhalo.ndim - 1))
    rvir = (mhalo / (rhoc * (4 / 3) * np.pi * Delc)) ** (1 / 3)
    return rvir.to(u.kpc)


def virial_T(mhalo, Rvir):
    if not isinstance(mhalo, u.Quantity):
        mhalo = mhalo * u.Msun
    if not isinstance(Rvir, u.Quantity):
        Rvir = Rvir * u.kpc
    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return ((2 / 5) * ((G * mhalo * mp) / (Rvir * kb))).to(u.K)


# %% observational comparison data, same as smhm_vcirc_over_redshift_plotting.py

# Shuntov et al. 2024 SMHM, per redshift bin (for the M_halo panel)
smf_data = Table.read("./data/Shuntov2024-shmr.ecsv", format="ascii.ecsv")
Mhalo_shuntov = {}
Mstar_shuntov = {}
Mstar_shuntov_low = {}
Mstar_shuntov_up = {}
for zb in zbins_str:
    Mhalo_shuntov[zb] = smf_data[smf_data["Redshift"] == zb]["M_halo"]
    Mstar_shuntov[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_50"]
    Mstar_shuntov_low[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_16"]
    Mstar_shuntov_up[zb] = smf_data[smf_data["Redshift"] == zb]["M_star_84"]

# Behroozi et al. 2019 z=0 SMHM (for the z=0 band in the M_halo panel)
smmr = np.loadtxt("./data/sm_averages_a1.002310.dat")
loghm_behroozi = smmr[5:21, 0]
logSMHM_behroozi = smmr[5:21, 10]
SMHMerr_behroozi = np.vstack([smmr[5:21, 12], smmr[5:21, 11]])
smhm_behroozi = 10**logSMHM_behroozi / (Ob0 / Omegam0)
smhm_behroozi_err_low = 10 ** SMHMerr_behroozi[0, :]
smhm_behroozi_err_up = 10 ** SMHMerr_behroozi[1, :]

# %% load the MULTIPLE_RUNS grid
file = "./runs/TEST_multiple_runs_smhm_vcirc_2phase_grid.h5"

with h5py.File(file, "r") as f:
    smhm = f["SMHM"][:]  # (n_z, n_mhalo, n_runs), already baryon-fraction normalized
    mhalo_obs = f["Mhalo_obs"][:] * u.Msun  # (n_z, n_mhalo, n_runs)
    zobs = f["redshifts"][:]
    n_runs = f.attrs["n_realizations"]
    start_times = f["start_times"][:]

print(f"loaded {file}: SMHM shape = {smhm.shape}, n_runs = {n_runs}")
print(f"start_times (Gyr) = {start_times}")

rvirs = virial_radius(zobs, mhalo_obs)
halo_Tvirs = virial_T(mhalo_obs, rvirs)

# summary stats across the N_HALOS_TO_RUN realizations (last axis), per
# (redshift, halo mass) grid point -- median line + 16th/84th percentile
# band. Percentiles (rather than median +/- std) keep the band strictly
# positive, which matters since these axes are log-scaled.
smhm_median = np.median(smhm, axis=-1)  # (n_z, n_mhalo)
smhm_lo = np.percentile(smhm, 16, axis=-1)
smhm_hi = np.percentile(smhm, 84, axis=-1)
# plain ndarrays (not Quantity): matplotlib's fill_between builds a raw
# float points array internally, and a Quantity with non-dimensionless
# units (K, Msun) raises a UnitConversionError when it gets implicitly
# cast to float there.
mhalo_median = np.median(mhalo_obs.value, axis=-1)  # Msun
Tvir_median = np.median(halo_Tvirs.value, axis=-1)  # K

# %% plot: "this work" column only (top: T_vir panel, bottom: M_halo panel)
fig, ax = plt.subplots(2, 1, figsize=(5.5, 9), dpi=150)

for i, zb in enumerate(zbins_str[::-1]):
    color = cmap_colors[::-1][i]

    # top row: SMHM vs T_vir -- median line + 16-84th percentile band
    # across the N_HALOS_TO_RUN realizations
    ax[0].fill_between(
        Tvir_median[i],
        smhm_lo[i],
        smhm_hi[i],
        color=color,
        alpha=0.25,
        lw=0,
        zorder=4,
    )
    ax[0].plot(
        Tvir_median[i],
        smhm_median[i],
        color=color,
        lw=2,
        zorder=6,
        label=f"${zbins_ctr[i]}$",
    )

    # bottom row: SMHM vs M_halo, plus the Shuntov+24 observational band
    ax[1].fill_between(
        Mhalo_shuntov[zb],
        np.array(Mstar_shuntov_low[zb]) / (np.array(Mhalo_shuntov[zb]) * (Ob0 / Omegam0)),
        np.array(Mstar_shuntov_up[zb]) / (np.array(Mhalo_shuntov[zb]) * (Ob0 / Omegam0)),
        lw=0,
        alpha=0.3,
        color=color,
        label=f"${zb}$",
    )
    ax[1].fill_between(
        mhalo_median[i],
        smhm_lo[i],
        smhm_hi[i],
        color=color,
        alpha=0.25,
        lw=0,
        zorder=4,
    )
    ax[1].plot(mhalo_median[i], smhm_median[i], color=color, lw=2, zorder=6)

# z = 0 bin, same grey treatment as the original script
ax[0].fill_between(
    Tvir_median[-1], smhm_lo[-1], smhm_hi[-1], color="grey", alpha=0.25, lw=0, zorder=4
)
ax[0].plot(Tvir_median[-1], smhm_median[-1], color="grey", lw=2, zorder=6, label="0.0")

ax[1].fill_between(
    mhalo_median[-1], smhm_lo[-1], smhm_hi[-1], color="grey", alpha=0.25, lw=0, zorder=4
)
ax[1].plot(mhalo_median[-1], smhm_median[-1], color="grey", lw=2, zorder=6)
ax[1].fill_between(
    10**loghm_behroozi,
    smhm_behroozi * smhm_behroozi_err_low,
    smhm_behroozi * smhm_behroozi_err_up,
    facecolor="grey",
    alpha=0.3,
    zorder=0,
)
ax[1].text(
    0.05,
    0.01,
    r"$z=0$ (Behroozi et al. 2019)",
    transform=ax[1].transAxes,
    fontsize=8,
    rotation=32,
    ha="left",
)

ax[0].set(
    xlabel=r"$T_{\rm vir} \: {\rm [K]}$",
    ylabel=r"$M_{\star}$/ $M_{\rm halo} (\Omega_{\rm b}/\Omega_{\rm m})$",
    yscale="log",
    xscale="log",
    xlim=(3e4, 5e7),
)
ax[1].set(
    xlabel=r"$M_{\rm halo} \: {\rm [M_\odot]}$",
    ylabel=r"$M_{\star}$/ $M_{\rm halo} (\Omega_{\rm b}/\Omega_{\rm m})$",
    yscale="log",
    xscale="log",
    xlim=(1e10, 9e12),
    ylim=(3e-3, 0.8),
)
ax[0].text(0.05, 0.95, "this work\n(MULTIPLE\\_RUNS)", transform=ax[0].transAxes, ha="left", va="top", fontsize=10)
ax[0].legend(frameon=False, fontsize=8, ncol=3, title=r"$z$", title_fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig("./runs/TEST_smhm_vcirc_multiple_runs_this_work.png", dpi=200, bbox_inches="tight")
plt.show()
