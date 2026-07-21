#XXX: maybe baseline model was ran with updated quantitities was ran to get the apper figure back?
# %%
import os
from pyexpat import model

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
from cgm_sf_regulator import CGMRegulator, mhalo_at_z0_fakhouri, halo_diagnostics_v2
from cgm_sf_regulator_baseline import CGMRegulatorBaseline
from run_grids_of_models import (
    run_baseline_model_redshift_grid,
    run_2phase_model_redshift_grid,
)
#%%

### update matplotlib settings
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
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
# %% run grid of models for comparison

# # write to file
# file = "./runs/smhm_baseline_z0p01-eta_elimited-M_10p0_0p7-E_0p1_0p4.h5"

# if not os.path.exists(file):
#     redshift_variation, zsims = run_baseline_model_redshift_grid(
#         observe_at=[0.01],  # redshift we want to observe
#         mhalos=[np.geomspace(1e10, 1e13, 6) * u.Msun],
#         write_to_file=file,
#     )
# else:
#     print("file already exists")
#     f = h5py.File(file, "r")
#     smhm_normalized = f["SMHM"][0]  # smhm is already normalized by baryon fractions
#     mhalo_obs = f["Mhalo_obs"][0]
#     print(f.keys())

# # %% newest model grid
# file_latest_model = "./runs/smhm_2phase_z0p01.h5"
# if not os.path.exists(file_latest_model):
#     redshift_variation, zsims = run_2phase_model_redshift_grid(
#         observe_at=[0.01],  # redshift we want to observe
#         mhalos=[np.geomspace(1e10, 1e13, 6) * u.Msun],
#         write_to_file=file_latest_model,
#     )
# else:
#     print("file already exists")
#     f = h5py.File(file_latest_model, "r")
#     smhm_2phase = f["SMHM"][0]
#     mhalo_obs_2phase = f["Mhalo_obs"][0]
    # print(f.keys())

# %% do a single run of each model
# mhalo_z0 = 1e12 * u.Msun
# t_span = (0.1, 13)  # gyrs

# #latest model
# latest_model = CGMRegulator(
#     mhalo_z0,
#     t_span,
# )
# run_latest = latest_model.run_halo()
# results_latest = latest_model.get_results()
# derived_latest = latest_model.get_derived_quantities()
# t_adaptive = results_latest["t"]


# #baseline model
# from cgm_sf_regulator_baseline import CGMRegulatorBaseline
# baseline_model = CGMRegulatorBaseline(
#     mhalo_z0,
#     time_interval=t_adaptive,  # use the same time array as the latest model for direct comparison
# )
# run = baseline_model.run_halo()
# results = baseline_model.get_results()
# derived = baseline_model.get_derived_quantities()



# %% get properties and plot them with the behroozi

# fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), dpi=300)
# plt.subplots_adjust(wspace=0.22)

# cmap = plt.get_cmap("Dark2")
# # Color assignments for clarity
# color_star = cmap(5)  # Stellar mass
# color_ism = cmap(4)  # ISM mass
# color_cgm = cmap(2)  # CGM mass
# color_baseline = cmap(0)  # Baseline model
# color_latest = cmap(1)  # Latest model

# ax[0].fill_between(
#     10**loghm,
#     smhm_behroozi * smhm_err_low,
#     smhm_behroozi * smhm_err_up,
#     color="grey",
#     alpha=0.5,
#     label="Behroozi+2019",
#     zorder=0,
# )

# ax[0].plot(mhalo_obs_2phase, smhm_2phase, label="this work", color=color_latest, lw=3)
# ax[0].plot(
#     mhalo_obs,
#     smhm_normalized,
#     label="baseline C23 model",
#     color=color_baseline,
#     lw=3,
#     ls="--",
# )
# ax[0].legend(loc="upper left", ncol=1, frameon=False, fontsize=10)
# ax[0].set(
#     xlabel=r"$M_{\rm halo}$ [M$_\odot$]",
#     ylabel=r"$M_{\star}$/ $M_{\rm halo} (\Omega_{\rm b}/\Omega_{\rm m})$",
#     yscale="log",
#     xscale="log",
#     xlim=(1e10, 2e12),
#     ylim=(1e-3, 1),
# )

# # add a text showing the redshift in the bottom right
# ax[0].text(
#     0.95,
#     0.05,
#     r"$z\sim0$",
#     transform=ax[0].transAxes,
#     ha="right",
#     va="bottom",
#     fontsize=12,
# )

# # make a custom legend, just dashed lines for stellar mass, ism mass and cgm mass
# ax[1].plot([], [], color=color_star, lw=2, ls="--", label=r" ")
# ax[1].plot([], [], color=color_ism, lw=2, ls="--", label=r" ")
# ax[1].plot([], [], color=color_cgm, lw=2, ls="--", label=r" ")

# # all the baseline models are dashed
# ax[1].plot(results["t"], results["m_star"], color=color_star, lw=2, ls="--")
# ax[1].plot(results["t"], results["m_gas"], color=color_ism, lw=2, ls="--")
# ax[1].plot(results["t"], results["m_cgm"], color=color_cgm, lw=2, ls="--")

# ax[1].plot(
#     results_latest["t"],
#     results_latest["m_star"],
#     color=color_star,
#     lw=2,
#     label=r"$M_\star$",
#     alpha=0.8,
# )
# ax[1].plot(
#     results_latest["t"],
#     results_latest["m_ism"],
#     color=color_ism,
#     lw=2,
#     label=r"$M_{\rm ISM}$",
#     alpha=0.8,
# )
# ax[1].plot(
#     results_latest["t"],
#     results_latest["m_cgm"],
#     color=color_cgm,
#     lw=2,
#     label=r"$M_{\rm CGM}$",
#     alpha=0.8,
# )

# # break the cgm which is colored purple ish into its hot and cold component for the latest model
# ax[1].plot(
#     results_latest["t"],
#     results_latest["m_cgm_hot"],
#     color="crimson",
#     lw=2,
#     label=r"$M_{\rm CGM, hot}$",
# )
# ax[1].plot(
#     results_latest["t"],
#     results_latest["m_cgm_cold"],
#     color="tab:blue",
#     lw=2,
#     label=r"$M_{\rm CGM, cold}$",
# )

# # add the baryon fraction
# ax[1].plot(results_latest["t"], results_latest["m_halo"] * f_b, color="grey", lw=2)
# ax[1].set(
#     ylabel=r"$\rm Masses ~ [{\rm M_{\odot}}]$",
#     yscale="log",
#     ylim=(1e7, None),
#     xlim=(0.25, results["t"][-1]),
#     xlabel=r"$t$ [Gyr]",
# )

# ax[1].legend(ncol=3, frameon=False, fontsize=9, loc="lower right")
# # add inline text legend for m baryon
# ax[1].text(
#     0.1,
#     0.95,
#     r"$f_{\rm b} M_{\rm halo}$",
#     transform=ax[1].transAxes,
#     ha="left",
#     va="top",
#     fontsize=10,
#     rotation=25,
#     color="grey",
# )
# ax[1].text(
#     0.23,
#     0.2,
#     r"baseline",
#     transform=ax[1].transAxes,
#     ha="center",
#     va="bottom",
#     fontsize=9,
#     color="k",
# )
# ax[1].text(
#     0.6,
#     0.2,
#     r"this work",
#     transform=ax[1].transAxes,
#     ha="center",
#     va="bottom",
#     fontsize=9,
#     color="k",
# )

# # now, do the same for  e_ism_wind e_cgm_cool  e_cgm_out  e_cgm_in  e_cgm
# # Plot energy flows for baseline (dashed) and latest (solid) models
# color_e_ism_wind = color_star
# color_e_cgm_cool = "tab:blue"
# color_e_cgm_out = color_ism
# color_e_cgm_in = color_cgm
# color_e_cgm = "tab:red"

# # make the dashed plots again for the legend()
# ax[2].plot([], [], color=color_e_ism_wind, lw=2, ls="--", label=r" ")
# ax[2].plot([], [], color=color_e_cgm_cool, lw=2, ls="--", label=r" ")
# ax[2].plot([], [], color=color_e_cgm_out, lw=2, ls="--", label=r" ")
# ax[2].plot([], [], color=color_e_cgm_in, lw=2, ls="--", label=r" ")
# ax[2].plot([], [], color=color_e_cgm, lw=2, ls="--", label=r" ")

# # Baseline model (dashed)
# ax[2].plot(results["t"], results["egy_ism_wind"], color=color_e_ism_wind, lw=2, ls="--")
# ax[2].plot(results["t"], results["egy_radloss"], color=color_e_cgm_cool, lw=2, ls="--")
# ax[2].plot(results["t"], results["egy_eject"], color=color_e_cgm_out, lw=2, ls="--")
# ax[2].plot(results["t"], results["egy_accrete"], color=color_e_cgm_in, lw=2, ls="--")
# ax[2].plot(results["t"], results["egy_cgm"], color=color_e_cgm, lw=2, ls="--")

# # Latest model (solid)
# ax[2].plot(
#     results_latest["t"],
#     results_latest["egy_ism_wind"],
#     color=color_e_ism_wind,
#     lw=2,
#     label=r"$E_{\rm SNe, wind}$",
#     alpha=0.8
# )
# ax[2].plot(
#     results_latest["t"],
#     results_latest["egy_radloss"],
#     color=color_e_cgm_cool,
#     lw=2,
#     label=r"$E_{\rm CGM, cool}$",
#     alpha=0.8
# )
# ax[2].plot(
#     results_latest["t"],
#     results_latest["egy_eject"],
#     color=color_e_cgm_out,
#     lw=2,
#     label=r"$E_{\rm CGM, ej}$",
#     alpha=0.8
# )
# ax[2].plot(
#     results_latest["t"],
#     results_latest["egy_cgm_in"],
#     color=color_e_cgm_in,
#     lw=2,
#     label=r"$E_{\rm CGM, acc}$",
#     alpha=0.8
# )
# ax[2].plot(
#     results_latest["t"],
#     results_latest["egy_cgm"],
#     color=color_e_cgm,
#     lw=2,
#     label=r"$E_{\rm CGM}$",
#     alpha=0.8
# )

# ax[2].set(
#     ylabel=r"$\rm Energies [{\rm erg}]$",
#     yscale="log",
#     ylim=(2e55, 6e58),
#     xlim=(0.25, results["t"][-1]),
#     xlabel=r"$t$ [Gyr]",
# )
# ax[2].legend(ncol=2, frameon=False, fontsize=9, loc="lower right")
# ax[2].text(
#     0.55,
#     0.35,
#     r"baseline",
#     transform=ax[2].transAxes,
#     ha="center",
#     va="bottom",
#     fontsize=9,
#     color="k",
# )
# ax[2].text(
#     0.8,
#     0.35,
#     r"this work",
#     transform=ax[2].transAxes,
#     ha="center",
#     va="bottom",
#     fontsize=9,
#     color="k",
# )

# # make a twin redshift axis for the top row, using z
# # get the current x axis labels of the first row and their
# # Get more ticks for the twin axis by interpolating between min and max time
# num_ticks = 6  # Increase number of ticks for finer resolution
# t_min = results["t"][0] * 0.99
# t_max = results["t"][-1]
# x_axis_tick_labels = np.linspace(t_min, t_max, num_ticks)

# for i in range(1, 3):
#     ax2 = ax[i].twiny()
#     # make sure the ranges are the same
#     ax2.plot(results["t"], results["m_star"], color="k", alpha=0)
#     ax2.set_xlim(t_min, t_max)
#     t_ticks = x_axis_tick_labels
#     z_ticks = cosmology.z_at_value(LCDM.age, t_ticks * u.Gyr).value
#     ax2.set_xticks(x_axis_tick_labels)
#     ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
#     ax2.set_xlim(t_min, t_max)
#     ax2.set_xlabel(r"$z$")
#     ax2.minorticks_off()
#     ax[i].minorticks_on()
# for axes in ax:
#     for line in axes.lines:
#         line.set_zorder(1)
# # plt.savefig("./final_figs/fig1_mw_mass_old.pdf", dpi=200, bbox_inches="tight", pad_inches=0.05)

# plt.show()
# %% new verions



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
# %% run grid of models for  SMHM comparison

# write to file
file = "./runs/smhm_baseline_z0p01-eta_elimited-M_10p0_0p7-E_0p1_0p4.h5"

if not os.path.exists(file):
    redshift_variation, zsims = run_baseline_model_redshift_grid(
        observe_at=[0.01],  # redshift we want to observe
        mhalos=[np.geomspace(1e10, 1e13, 6) * u.Msun],
        write_to_file=file,
    )
else:
    print("file already exists")
    f = h5py.File(file, "r")
    smhm_normalized = f["SMHM"][0]  # smhm is already normalized by baryon fractions
    mhalo_obs = f["Mhalo_obs"][0]
    print(f.keys())

# latest model grid
file_latest_model = "./runs/redshift_scan_KS_1998_kappa_updated_0p02_n1p8_r0p018_etaZ0p7_alphaE0p1_alphaM0p1.h5"
if not os.path.exists(file_latest_model):
    redshift_variation, zsims = run_2phase_model_redshift_grid(
        observe_at=[0.01],  # redshift we want to observe
        mhalos=[np.geomspace(1e10, 1e13, 6) * u.Msun],
        write_to_file=file_latest_model,
        KS_kappa_s=0.02,
        KS_n=1.8,
        disk_scale_length=0.018,
        KS_parametrization="KS1998",
    )
else:
    print("file already exists")
    f = h5py.File(file_latest_model, "r")
    smhm_2phase = f["SMHM"][-1]
    mhalo_obs_2phase = f["Mhalo_obs"][-1]
    print(f.keys())

# %% do a single run of each model for 1to 1 comparison
mhalo_z0 = 1e12 * u.Msun
t_span = (0.1, 13)  # gyrs

latest_model = CGMRegulator(
    mhalo_z0,
    t_span,
    add_f_prevent_floor=1e-8,# virtually no floor
    KS_kappa_s = 0.02, # fiducial params
    KS_n = 1.8,
    disk_scale_length=0.018,
    KS_parametrization="KS1998",
    TEST_tej_Tvir_definition=False,
    eta_z=0.6,
    alpha_e=0.1,
    alpha_m=0.1,
)
run_latest = latest_model.run_halo()
results_latest = latest_model.get_results()
derived_latest = latest_model.get_derived_quantities()
t_adaptive = results_latest["t"]

#%% STIFFNESS CHECK
stiff = latest_model.quantify_stiffness(
    n_samples=8,
    include_mode_sources=True,
    top_n=4,
)

print("max stiffness ratio:", stiff["max_stiffness_ratio"])
print("median stiffness ratio:", stiff["median_stiffness_ratio"])

for s in stiff["samples"]:
    print(
        s["t_gyr"],
        s["stiffness_ratio"],
        s["fastest_timescale_gyr"],
        s.get("dominant_variables", []),
    )

#%%
baseline_model = CGMRegulatorBaseline(
    mhalo_z0,
    time_interval=t_adaptive,  # use the same time array as the latest model for direct comparison
)
run = baseline_model.run_halo()
results = baseline_model.get_results()
derived = baseline_model.get_derived_quantities()
#%%
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

cmap = plt.get_cmap("Dark2")
# Color assignments for clarity
color_star = cmap(5)  # Stellar mass
color_ism = cmap(4)  # ISM mass
color_cgm = "grey"# cmap(2)  # CGM mass
color_baseline = cmap(0)  # Baseline model
color_latest = cmap(1)  # Latest model


fig, ax = plt.subplots(5, 2, figsize=(12, 12.25), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.42, wspace=0.15)
ax = ax.flatten()
# make a inset panel to take its place
ax0 = ax[0].inset_axes([0, -1.3, 1, 2.3])

ax0.fill_between(
    10**loghm,
    smhm_behroozi * smhm_err_low,
    smhm_behroozi * smhm_err_up,
    color="grey",
    alpha=0.5,
    zorder=0,
)

ax0.plot(mhalo_obs_2phase, smhm_2phase, label="this work", color=color_latest, lw=3)
ax0.plot(
    mhalo_obs,
    smhm_normalized,
    label="C23 model",
    color=color_baseline,
    lw=3,
    ls="--",
)
ax0.legend(
    loc="upper center", ncol=3, frameon=False, fontsize=11
)
ax0.set(
    xlabel=r"$M_{\rm halo}$ [M$_\odot$]",
    ylabel=r"$M_{\star}$/ $M_{\rm halo} (\Omega_{\rm b}/\Omega_{\rm m})$",
    yscale="log",
    xscale="log",
    xlim=(1e10, 2e12),
    ylim=(1e-3, 0.3),
)
# add the behroozi in panel
ax0.text(0.01,0.15, "Behroozi et. al. 2019", transform=ax0.transAxes, rotation=37, va="bottom", ha="left", fontsize=10)

# add a text showing the redshift in the bottom right
ax0.text(
    0.95,
    0.05,
    r"$z\sim0$",
    transform=ax0.transAxes,
    ha="right",
    va="bottom",
    fontsize=12,
)

# all the baseline models are dashed
ax[4].plot(results_latest["t"], results_latest["m_star"], color=color_latest, lw=3)
ax[4].plot(results["t"], results["m_star"], color=color_baseline, lw=3, ls="--")
ax[4].set(
    ylabel=r"$ M_{\star} ~ [{\rm M_{\odot}}]$",
    yscale="log",
    ylim=(2e6, None),
    xlim=(0.25, results["t"][-1]),
    
)
# add the mass of this halo
ax[4].text(0.15, 0.85, r"$M_{\rm halo} = 10^{12} ~{\rm M_{\odot}}$", transform=ax[4].transAxes, ha="left", va="top", fontsize=12)


# make an inset to plot the fractional difference
ax4 = ax[4].inset_axes([0, -0.3, 1, 0.3])
ax4.plot(results_latest["t"], (results_latest["m_star"] - results["m_star"]) / results["m_star"], color=color_cgm , lw=1.5)
ax4.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax4.axhline(0, color="k", lw=0.5, ls=":")
ax4.set_ylabel(ylabel="frac. diff", fontsize=9)
#remove tick labels for x axis
ax4.set_xticklabels([])


ax[6].plot(results_latest["t"], results_latest["m_ism"], color=color_latest, lw=3)
ax[6].plot(results["t"], results["m_gas"], color=color_baseline, lw=3, ls="--")
ax[6].set(
    ylabel=r"$ M_{\rm ISM} ~ [{\rm M_{\odot}}]$",
    yscale="log",
    ylim=(2e6, None),
    xlim=(0.25, results["t"][-1]),
)
ax6 = ax[6].inset_axes([0, -0.3, 1, 0.3])
ax6.plot(results_latest["t"], (results_latest["m_ism"] - results["m_gas"]) / results["m_gas"], color=color_cgm  , lw=1.5)
ax6.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax6.axhline(0, color="k", lw=0.5, ls=":")
ax6.set_ylabel(ylabel="frac. diff", fontsize=9)
ax6.set_xticklabels([])

ax[8].plot(
    results_latest["t"],
    results_latest["m_cgm"],
    color=color_latest,
    lw=3,
)
ax[8].plot(results["t"], results["m_cgm"], color=color_baseline, lw=3, ls="--")
ax[8].set(
    ylabel=r"$ M_{\rm CGM} ~ [{\rm M_{\odot}}]$",
    yscale="log",
    ylim=(2e6, None),
    xlim=(0.25, results["t"][-1]),

)
ax8 = ax[8].inset_axes([0, -0.3, 1, 0.3])
ax8.plot(results_latest["t"], (results_latest["m_cgm"] - results["m_cgm"]) / results["m_cgm"], color=color_cgm , lw=1.5)
ax8.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax8.axhline(0, color="k", lw=0.5, ls=":")
ax8.set_ylabel(ylabel="frac. diff", fontsize=9)
ax8.set_xlabel(r"$t$ [Gyr]")


# show the cool and hot CGM phases
ax[8].plot(
    results_latest["t"],
    results_latest["m_cgm_hot"],
    color=cmap(5),
    lw=2,
    label="hot CGM",
)
ax[8].plot(
    results_latest["t"],
    results_latest["m_cgm_cold"],
    color="tab:blue",
    lw=2,
    label="cool CGM",
)
ax[8].legend(frameon=False, fontsize=12, loc="lower right")

# total energy
ax[1].plot(results_latest["t"], results_latest["egy_cgm"], color=color_latest, lw=3)
ax[1].plot(results["t"], results["egy_cgm"], color=color_baseline, lw=3, ls="--")

ax[1].set(
    ylabel=r"total $E_{\rm CGM}$",
    yscale="log",
    ylim=(2e54, 6e58),
    xlim=(0.25, results["t"][-1]),
)
ax1 = ax[1].inset_axes([0, -0.3, 1, 0.3])
ax1.plot(results_latest["t"], (results_latest["egy_cgm"] - results["egy_cgm"]) / results["egy_cgm"], color=color_cgm , lw=1.5)
ax1.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax1.axhline(0, color="k", lw=0.5, ls=":")
ax1.set_ylabel(ylabel="frac. diff", fontsize=9)
ax1.set_xticklabels([])

# SNE winds energy
ax[3].plot(
    results_latest["t"], results_latest["egy_ism_wind"], color=color_latest, lw=3
)
ax[3].plot(results["t"], results["egy_ism_wind"], color=color_baseline, lw=3, ls="--")
ax[3].set(
    ylabel=r"$E_{\rm SNe, wind}$",
    yscale="log",
    ylim=(2e54, 6e58),
    xlim=(0.25, results["t"][-1]),
)
ax3 = ax[3].inset_axes([0, -0.3, 1, 0.3])
ax3.plot(results_latest["t"], (results_latest["egy_ism_wind"] - results["egy_ism_wind"]) / results["egy_ism_wind"], color=color_cgm  , lw=1.5)
ax3.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax3.axhline(0, color="k", lw=0.5, ls=":")
ax3.set_ylabel(ylabel="frac. diff", fontsize=9)
ax3.set_xticklabels([])

# cooling
ax[5].plot(results_latest["t"], results_latest["egy_radloss"], color=color_latest, lw=3)
ax[5].plot(results["t"], results["egy_radloss"], color=color_baseline, lw=3, ls="--")
ax[5].set(
    ylabel=r"$E_{\rm cooling}$",
    yscale="log",
    ylim=(2e54, 6e58),
    xlim=(0.25, results["t"][-1]),
)
ax5 = ax[5].inset_axes([0, -0.3, 1, 0.3])
ax5.plot(results_latest["t"], (results_latest["egy_radloss"] - results["egy_radloss"]) / results["egy_radloss"], color=color_cgm  , lw=1.5)
ax5.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax5.axhline(0, color="k", lw=0.5, ls=":")
ax5.set_ylabel(ylabel="frac. diff", fontsize=9)
ax5.set_xticklabels([]) 

# ejecting
ax[7].plot(results_latest["t"], results_latest["egy_eject"], color=color_latest, lw=3)
ax[7].plot(results["t"], results["egy_eject"], color=color_baseline, lw=3, ls="--")
ax[7].set(
    ylabel=r"$E_{\rm ej}$",
    yscale="log",
    ylim=(2e54, 6e58),
    xlim=(0.25, results["t"][-1]),
)
ax7 = ax[7].inset_axes([0, -0.3, 1, 0.3])
ax7.plot(results_latest["t"], (results_latest["egy_eject"] - results["egy_eject"]) / results["egy_eject"], color=color_cgm , lw=1.5)
ax7.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax7.axhline(0, color="k", lw=0.5, ls=":")
ax7.set_ylabel(ylabel="frac. diff", fontsize=9)
ax7.set_xticklabels([])

# accreting
ax[9].plot(results_latest["t"], results_latest["egy_cgm_in"], color=color_latest, lw=3)
ax[9].plot(results["t"], results["egy_accrete"], color=color_baseline, lw=3, ls="--")
ax[9].set(
    ylabel=r"$E_{\rm in}$",
    yscale="log",
    ylim=(2e54, 6e58),
    xlim=(0.25, results["t"][-1]),
    xscale="log",
)
ax9 = ax[9].inset_axes([0, -0.3, 1, 0.3])
ax9.plot(results_latest["t"], (results_latest["egy_cgm_in"] - results["egy_accrete"]) / results["egy_accrete"], color=color_cgm , lw=1.5)
ax9.set( xlim=(0.25, results["t"][-1]), ylim=(-1.5, 1.5), xscale="log")
ax9.axhline(0, color="k", lw=0.5, ls=":")
ax9.set_ylabel(ylabel="frac. diff", fontsize=9)
ax9.set_xlabel(r"$t$ [Gyr]")

# Apply the same settings to all relevant axes except ax[0] and ax[2] (which are turned off)
for i in range(len(ax)):
    if i in [0, 2]:
        continue
    ax[i].set(
        xscale="log",
        xlim=(0.25, results["t"][-1])
    )


# make a twin redshift axis for the top row, using z
# get the current x axis labels of the first row and their
# Get more ticks for the twin axis by interpolating between min and max time
# Choose exact ticks for time axis (in Gyr) and corresponding redshifts
t_ticks = np.array([0.3, 0.5, 1, 2, 4, 10, ])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]

for i, a in enumerate(ax):
    if (i == 0) or (i == 2):
        continue
    ax2 = ax[i].twiny()
    # dummy plot to sync axis
    ax2.plot(results["t"], results["m_star"], color="k", alpha=0)
    ax2.set(xscale="log", xlim=(0.25, results["t"][-1]))
    if (i == 1) or (i == 4):
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks], fontsize=10)
        ax2.set_xlabel(r"$z$", fontsize=11)
    else:
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels([])
  
    ax2.minorticks_off()
    ax[i].minorticks_on()


for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)

# turn off panel 0 and 2
ax[0].axis("off")
ax[2].axis("off")
# Add panel labels (a), (b), ...
panel_labels = ["", "(a)", "", "(b)",  "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]
for i, axes in enumerate(ax):
    # Place label in upper left, skip panels that are turned off
    if i not in [0, 2]:
        axes.text(
            0.05, 0.90, panel_labels[i],
            transform=axes.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
        # turn on minor ticks for all axes
        axes.minorticks_on()

plt.savefig("./final_figs/fig_1_integrated_mw_mass.pdf", dpi=200, bbox_inches="tight", pad_inches=0.05)

plt.show()
# %%

