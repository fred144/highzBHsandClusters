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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from run_grids_of_models import (
    run_baseline_model_redshift_grid,
    run_2phase_model_redshift_grid,
)
import matplotlib.lines as mlines

### update matplotlib settings
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": False,
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

# %% Plot timescales for baseline and latest (2-phase) models in a single figure

mhalo_z0 = 1e12 * u.Msun
t_span = (0.1, 13)  # gyrs

# Baseline model
baseline_model = CGMRegulatorBaseline(
    mhalo_z0,
    t_span,
    tstep=0.0011,
    updated_halo_infall=False,
    updated_loadings=False,
    updated_SF_law=False,
    verbose=False,
)
baseline_model.run_halo()
results_baseline = baseline_model.get_results()
derived_baseline = baseline_model.get_derived_quantities()

# 2-phase (latest) model
model_2phase = CGMRegulator(
    mhalo_z0,
    t_span,
    KS_kappa_s=0.1,
    add_f_prevent_floor=0.01,
    verbose=False,
    dbug_norm_for_2_phase_CGM=0,
)
model_2phase.run_halo()
results_2phase = model_2phase.get_results()
derived_2phase = model_2phase.get_derived_quantities()
# %%
cmap = plt.get_cmap("Dark2")
color_star = "darkorange"
color_green = "tab:green"
color_cgm = "dodgerblue"
color_baseline = cmap(0)
color_latest = cmap(1)

fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.1)

# --- Top panel: Baseline model timescales ---
t_cool = derived_baseline["tcool_real"]
t_dynamical = derived_baseline["t_dyn"]
t_cool_eff = derived_baseline["tcool_eff"]
t_dep = derived_baseline["t_dep"]
t_ejection = derived_baseline["t_ejection"]
sim_time = derived_baseline["sim_time"]

# ax[0].plot(sim_time, t_cool,|lw=3, color=color_cgm, label=r"$t_{\mathrm{cool}}$")

ax.plot(
    sim_time,
    t_cool_eff,
    lw=3,
    color=color_cgm,
    label=r"$t_{\mathrm{cool,eff}}$",
    alpha=0.6,
)
ax.plot(sim_time, t_dep, lw=3, color=color_star, label=r"$t_{\mathrm{dep}}$", alpha=0.6)
ax.plot(
    sim_time,
    t_ejection,
    lw=3,
    color=color_green,
    label=r"$t_{\mathrm{ejection}}$",
    ls=":",
    alpha=0.8,
)

ax.plot(
    sim_time, t_dynamical, lw=3, color="grey", label=r"$t_{\mathrm{ff}}$", alpha=0.6
)
# ax[0].set(
#     yscale="log",
#     xscale="log",
#     ylim=(1.5e-3, 13),
#     xlim=(results_baseline["t"][0] * 0.8, 13),
#     ylabel=r"timescales $[\mathrm{Gyr}]$",
# )
# ax[0].legend(frameon=False, ncol=2, fontsize=10, title="baseline model")

# Twin redshift axis for top panel
t_ticks = np.array([0.3, 0.5, 1, 2, 4, 8, 12])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
ax2_top = ax.twiny()
ax2_top.plot(results_baseline["t"], results_baseline["m_star"], color="k", alpha=0)
ax2_top.set(
    xscale="log",
    xlim=(0.25, results_baseline["t"][-1]),
)
ax2_top.set_xticks(t_ticks)
ax2_top.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
ax2_top.set_xlabel(r"$z$", labelpad=8)
ax2_top.minorticks_off()
ax.minorticks_on()


t_cool_2 = derived_2phase["tcool_real"]
t_dynamical_2 = derived_2phase["t_dynamical"]
t_dep_effect = results_2phase["m_star"] / derived_2phase["dot_m_sfr"]
t_ejection_2 = derived_2phase["t_ejection"]
sim_time_2 = derived_2phase["sim_time"]

ax.plot(sim_time_2, t_cool_2, lw=3, color=color_cgm, label=r"$t_{\mathrm{cool}}$")
# ax.plot(sim_time_2, t_dynamical_2, lw=3, color="grey", label=r"$t_{\mathrm{ff}}$")
ax.plot(
    sim_time_2, t_dep_effect, lw=3, color=color_star, label=r"$t_{\mathrm{dep, eff}}$"
)
ax.plot(
    sim_time_2,
    t_ejection_2,
    lw=3,
    color=color_green,
    label=r"$t_{\mathrm{ejection}}$",
    ls="--",
)
# Overlay baseline depletion time for comparison
# ax[1].plot(sim_time, t_dep, lw=3, color=color_star, label=r"$t_{\mathrm{dep}}$", ls="--")

ax.set(
    yscale="log",
    xscale="log",
    ylim=(1e-3, 13),
    xlim=(results_2phase["t"][0] * 0.8, 13),
    ylabel=r"timescales $[\mathrm{Gyr}]$",
)
ax.legend(frameon=False, ncol=2, fontsize=10)

ax.set_xlabel(r"time $[\mathrm{Gyr}]$")

# Share x-axis between panels
# ax[1].set_xlim(ax[0].get_xlim())

# plt.savefig(
#     "./figures/model_timescales_comparison.png",
#     dpi=200,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
plt.show()

# %% let's look in more detail where the oscillations originate for the 2 phase model

# Define geometrically spaced halo masses
halo_masses = np.logspace(10, 13, 12) * u.Msun  # You can change the range and number
# %%
# store results for each halo mass\\
halo_results = []
for mhalo in halo_masses:
    model_2phase = CGMRegulator(
        mhalo, time_interval=(0.1, 13.3), KS_kappa_s=0.1, add_f_prevent_floor=1e-6, dbug_norm_for_2_phase_CGM=0, verbose=False
    )
    model_2phase.run_halo()
    results_2phase = model_2phase.get_results()
    derived_2phase = model_2phase.get_derived_quantities()
    sim_time_2 = derived_2phase["sim_time"]
    t_cool_2 = derived_2phase["tcool_real"]
    t_dep_effect = results_2phase["m_star"] / derived_2phase["dot_m_sfr"]
    t_ejection_2 = derived_2phase["t_ejection"]
    t_ff_2 = derived_2phase["t_dynamical"]

    halo_vir_temp = derived_2phase["halo_vir_temp"]
    cgm_temp = derived_2phase["cgm_temp"]
    rho0 = derived_2phase["rho0"]
    dot_e_cgm_out = derived_2phase["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_2phase["dot_e_cgm_cooling"]
    dot_e_cgm_in = derived_2phase["dot_e_cgm_in"]
    dot_e_ism_wind = derived_2phase["dot_e_ism_wind"]

    dot_e_plus = dot_e_cgm_in + dot_e_ism_wind
    dot_e_minus = dot_e_cgm_cooling + dot_e_cgm_cooling
    # metal evolution
    m_metals = results_2phase["m_metals"]
    cgm_metallicity = m_metals / (
        results_2phase["m_cgm_hot"] + results_2phase["m_cgm_cold"]
    )
    cgm_Z_sun = cgm_metallicity / 0.0134

    halo_results.append(
        {
            "mhalo": mhalo,
            "sim_time_2": sim_time_2,
            "t_cool_2": t_cool_2,
            "t_dep_effect": t_dep_effect,
            "t_ejection_2": t_ejection_2,
            "t_vir": halo_vir_temp,
            "t_ff_2": t_ff_2,
            "cgm_temp": cgm_temp,
            "rho0": rho0,
            "dot_e_plus": dot_e_plus,
            "dot_e_minus": dot_e_minus,
            "dot_e_cgm_out": dot_e_cgm_out,
            "dot_e_cgm_in": dot_e_cgm_in,
            "dot_e_cgm_cooling": dot_e_cgm_cooling,
            "dot_e_ism_wind": dot_e_ism_wind,
            "cgm_Z_sun": cgm_Z_sun,
        }
    )
# %% results for each halo mass for the baseline model
halo_results_baseline = []

for mhalo in halo_masses:
    baseline_model = CGMRegulatorBaseline(
        mhalo,
        time_interval=(0.1, 13.3),
        updated_halo_infall=False,
        updated_loadings=False,
        updated_SF_law=False,
        # tstep=0.0001,
    )
    baseline_model.run_halo()
    results_baseline = baseline_model.get_results()
    derived_baseline = baseline_model.get_derived_quantities()
    sim_time = derived_baseline["sim_time"]
    t_cool_eff = derived_baseline["tcool_eff"]
    t_dep = derived_baseline["t_dep"]
    t_ejection = derived_baseline["t_ejection"]
    t_ff = derived_baseline["t_dyn"]

    halo_vir_t = derived_baseline["tvir"]

    dot_e_cgm_out = derived_baseline["dot_e_cgm_out"]
    dot_e_cgm_cooling = derived_baseline["dot_e_cgm_cool"]
    dot_e_cgm_in = derived_baseline["dot_e_cgm_in"]
    dot_e_ism_wind = derived_baseline["dot_e_ism_wind"]

    halo_results_baseline.append(
        {
            "mhalo": mhalo,
            "sim_time": sim_time,
            "t_cool_eff": t_cool_eff,
            "t_dep": t_dep,
            "t_ejection": t_ejection,
            "t_vir": halo_vir_t,
            "t_ff": t_ff,
            "dot_e_cgm_out": dot_e_cgm_out,
            "dot_e_cgm_cooling": dot_e_cgm_cooling,
            "dot_e_cgm_in": dot_e_cgm_in,
            "dot_e_ism_wind": dot_e_ism_wind,
        }
    )
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300, sharex=True)
ax.plot(
    derived_baseline["sim_time"],
    results_baseline["egy_cgm"],
    lw=3,
    color="grey",
    label=r"$E_{\mathrm{CGM}}$",
)
ax.set(yscale="log")
plt.show()
# %% plot of the timescale ratios using the stored results

# colormap
cmap = matplotlib.cm.get_cmap("coolwarm")  # cmr.tropical_r

# segmented the colormap based on the number of halo samples
num_halos = len(halo_masses)
cmap_segmented = matplotlib.cm.get_cmap("coolwarm", num_halos)
colors = [cmap_segmented(i) for i in range(num_halos)]
norm = matplotlib.colors.LogNorm(
    vmin=halo_masses.value.min(), vmax=halo_masses.value.max()
)
sm = plt.cm.ScalarMappable(cmap=cmap_segmented, norm=norm)
#%%
fig, ax = plt.subplots(3, 2, figsize=(9.5, 7.5), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.05)

ax = ax.flatten()

threshold = 1e6  # K, example threshold for virial temperature

for res in halo_results:
    color = cmap(norm(res["mhalo"].value))
    # Since matplotlib does not support per-point linestyle, split into segments
    times = res["sim_time_2"]
    y_dep = res["t_dep_effect"] / res["t_cool_2"]
    y_eject = res["t_ejection_2"] / res["t_cool_2"]
    y_tff = res["t_ff_2"] / res["t_cool_2"]
    t_vir = res["t_vir"]
    t_cgm = res["cgm_temp"]
    mask = t_vir > threshold
    if np.sum(mask) > 0:
        ax[0].plot(times[mask], y_dep[mask], ls="--", lw=2, color=color)
        ax[4].plot(times[mask], y_eject[mask], ls="--", lw=2, color=color)
        ax[2].plot(times[mask], y_tff[mask], ls="--", lw=2, color=color)

        ax[0].plot(times[~mask], y_dep[~mask], ls="-", lw=2, color=color)
        ax[4].plot(times[~mask], y_eject[~mask], ls="-", lw=2, color=color)
        ax[2].plot(times[~mask], y_tff[~mask], ls="-", lw=2, color=color)

    else:
        ax[0].plot(times, y_dep, ls="-", lw=2, alpha=0.9, color=color)
        ax[4].plot(times, y_eject, ls="-", lw=2, alpha=0.9, color=color)
        ax[2].plot(times, y_tff, ls="-", lw=2, alpha=0.9, color=color)

ax[0].set(
    yscale="log",
    xscale="log",
    xlim=(halo_results[0]["sim_time_2"][0], 2),
    ylabel=r"$t_{\mathrm{dep, eff}}/t_{\mathrm{cool}}$",
    ylim=(1e-2, 1e4),
)
# ax[0].axhline(0.06)
# ax[2].axhline(0.06)
ax[4].set(
    yscale="log",
    xscale="log",
    xlim=(halo_results[0]["sim_time_2"][0], 2),
    ylabel=r"$t_{\mathrm{ejection}}/t_{\mathrm{cool}}$",
    ylim=(2e-3, 500),
    xlabel=r"$t_{\rm univ} [\mathrm{Gyr}]$",
)
ax[2].set(
    yscale="log",
    xscale="log",
    xlim=(halo_results[0]["sim_time_2"][0], 2),
    ylabel=r"$t_{\mathrm{ff}}/t_{\mathrm{cool}}$",
    ylim=(2e-3, 500),
)

for res in halo_results_baseline:
    color = cmap(norm(res["mhalo"].value))

    t_vir = res["t_vir"]
    y_dep = res["t_dep"] / res["t_cool_eff"]
    y_eject = res["t_ejection"] / res["t_cool_eff"]
    y_tff = res["t_ff"] / res["t_cool_eff"]
    times = res["sim_time"]
    mask = t_vir > threshold
    if np.sum(mask) > 0:
        ax[1].plot(times[mask], y_dep[mask], ls="--", lw=2, color=color)
        ax[5].plot(times[mask], y_eject[mask], ls="--", lw=2, color=color)
        ax[3].plot(times[mask], y_tff[mask], ls="--", lw=2, color=color)

        ax[1].plot(times[~mask], y_dep[~mask], ls="-", lw=2, color=color)
        ax[5].plot(times[~mask], y_eject[~mask], ls="-", lw=2, color=color)
        ax[3].plot(times[~mask], y_tff[~mask], ls="-", lw=2, color=color)
    else:
        ax[1].plot(res["sim_time"], y_dep, color=color, alpha=0.5)
        ax[5].plot(res["sim_time"], y_eject, color=color, alpha=0.5)
        ax[3].plot(res["sim_time"], y_tff, color=color, alpha=0.5)

ax[1].set(
    yscale="log",
    xscale="log",
    xlim=(halo_results_baseline[0]["sim_time"][0] * 0.8, 5),
    ylabel=r"$t_{\mathrm{dep}}/t_{\mathrm{cool, eff}}$",
    ylim=(0.8, 10),
)
ax[5].set(
    yscale="log",
    xscale="log",
    xlim=(halo_results_baseline[0]["sim_time"][0] * 0.8, 5),
    ylabel=r"$t_{\mathrm{ejection}}/t_{\mathrm{cool, eff}}$",
    ylim=(0.1, 2),
    xlabel=r"$t_{\rm univ} [\mathrm{Gyr}]$",
)
ax[2].set(
    yscale="log",
    xscale="log",
    ylabel=r"$t_{\mathrm{ff}}/t_{\mathrm{cool}}$",
    ylim=(1e-2, 5e2),
)
ax[3].set(
    yscale="log",
    xscale="log",
    xlim=(halo_results_baseline[0]["sim_time"][0] * 0.8, 5),
    ylabel=r"$t_{\mathrm{ff}}/t_{\mathrm{cool, eff}}$",
    ylim=(0.1, 2),
)
# # Mark the inset regions on the main axes
# mark_inset(ax[2], inset_ax2, loc1=1, loc2=3,  lw=1)
# mark_inset(ax[3], inset_ax3, loc1=1, loc2=3, lw=1)

# add colorbar for halo mass (log scale)
cbar_ax = ax[1].inset_axes([0.1, 0.3, 0.6, 0.08])  # [x, y, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")

cbar.set_label(r"$M_{\rm halo} (z = 0) \ [{\rm M_\odot}]$")
cbar.set_ticks(halo_masses.value)
# cbar.set_ticklabels([f"{m:.1e}" for m in halo_masses.value])
cbar.ax.set_xscale("log")

# add twin redshift axis for the top row of plots (indices 0 and 1)
for i in [0, 1]:
    t_ticks = np.array([0.14, 0.3, 0.5, 1, 2, 3, 5])
    z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
    ax2 = ax[i].twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax[i].get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    ax2.set_xlabel(r"$z$", labelpad=8)
    ax2.minorticks_off()
    ax[i].minorticks_on()
for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)

# add a refence line for y = 1
ax[0].axhline(1, ls="-", lw=2, color="grey", alpha=0.5)

ax[1].axhline(1, ls="-", lw=2, color="grey", alpha=0.5)
ax[2].axhline(1, ls="-", lw=2, color="grey", alpha=0.5)
ax[3].axhline(1, ls="-", lw=2, color="grey", alpha=0.5)
ax[4].axhline(1, ls="-", lw=2, color="grey", alpha=0.5)
ax[5].axhline(1, ls="-", lw=2, color="grey", alpha=0.5)

# add text
ax[0].text(
    0.90,
    0.85,
    "this work",
    transform=ax[0].transAxes,
    fontsize=12,
    ha="right",
    va="top",
)
ax[1].text(
    0.90,
    0.85,
    "baseline C23 model",
    transform=ax[1].transAxes,
    fontsize=12,
    ha="right",
    va="top",
)


# plt.savefig(
#     "./figures/timescale_ratios.png", dpi=200, bbox_inches="tight", pad_inches=0.05
# )
plt.show()

# %% density and temperature of CGM
fig, ax = plt.subplots(4, 2, dpi=300, figsize=(4.25, 8.5), sharex="col", sharey="row")
ax = ax.flatten()
plt.subplots_adjust(hspace=0.065, wspace=0.1)

for res in halo_results:
    color = cmap(norm(res["mhalo"].value))
    times = res["sim_time_2"]
    t_vir = res["t_vir"]
    t_cgm = res["cgm_temp"]
    rho0 = res["rho0"]
    Z_sun = res["cgm_Z_sun"]

    # First row: CGM temperature
    ax[0].plot(times, t_cgm, color=color)
    ax[1].plot(times, t_cgm, color=color)

    # Second row: Virial temperature
    ax[2].plot(times, t_vir, color=color)
    ax[3].plot(times, t_vir, color=color)

    # third row: rho0
    ax[4].plot(times, rho0, color=color)
    ax[5].plot(times, rho0, color=color)

    # fourth row: Zsun
    ax[6].plot(times, Z_sun, color=color)
    ax[7].plot(times, Z_sun, color=color)

    # ax[5].set(yscale="log", xlim=(8.05, 13))
ax[0].set(
    yscale="log",
    ylabel=r"$T_{\rm CGM} [K]$",
    ylim=(2e4, 1e7),
)
ax[1].set(yscale="log", xlim=(8.05, 13), ylim=(2e4, 1e7))
ax[2].set(yscale="log", xlim=(0.05, 1.05), ylabel=r"$T_{\rm vir}$ [K]", ylim=(2e4, 1e8))
ax[3].set(yscale="log", xlim=(8.05, 13), ylim=(2e4, 1e7))

ax[4].set(
    yscale="log",
    xlim=(0.05, 1.05),
    ylabel=r"$\rho_{\rm 0,hot} ~ [{\rm M_\odot ~kpc^{-3}}]$",
)
ax[5].set(yscale="log", xlim=(8.05, 13), ylim=(200, 3e7))
ax[6].set(yscale="log", xlim=(0.05, 1.05), ylabel=r"$Z_{\rm CGM} [{\rm Z_{\odot}}]$")


# Add twin redshift axis for the top row of plots (indices 0 and 1)
ticks = [np.array([0.15, 0.4, 0.8]), np.array([8.5, 10, 12, 13])]
for i in [0, 1]:
    z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in ticks[i]]
    ax2 = ax[i].twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax[i].get_xlim())
    ax2.set_xticks(ticks[i])
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    ax2.minorticks_off()
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax[i].minorticks_on()

# turn off right ticks and spines for the left column and the same with the left spines for the right column
for i in [0, 2, 4, 6]:
    ax[i].yaxis.tick_left()
    ax[i].spines["right"].set_visible(False)
for i in [1, 3, 5, 7]:
    ax[i].yaxis.tick_right()
    ax[i].spines["left"].set_visible(False)

d = 0.95  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=5,
    linestyle="none",
    color="k",
    mec="k",
    mew=0.8,
    clip_on=False,
)
ax[0].plot([1, 1], [1, 0], transform=ax[0].transAxes, **kwargs)
ax[2].plot([1, 1], [1, 0], transform=ax[2].transAxes, **kwargs)
ax[4].plot([1, 1], [1, 0], transform=ax[4].transAxes, **kwargs)
ax[6].plot([1, 1], [1, 0], transform=ax[6].transAxes, **kwargs)

ax[1].plot([0, 0], [1, 0], transform=ax[1].transAxes, **kwargs)
ax[3].plot([0, 0], [1, 0], transform=ax[3].transAxes, **kwargs)
ax[5].plot([0, 0], [1, 0], transform=ax[5].transAxes, **kwargs)
ax[7].plot([0, 0], [1, 0], transform=ax[7].transAxes, **kwargs)
# Add colorbar for halo mass (log scale) on the right using inset axes
cbar_ax = ax[1].inset_axes([-0.25, 0.3, 1, 0.06], zorder=10)  # [x, y, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label(r"$M_{\rm halo} (z=0) \ [M_\odot]$")

# label for the figure
fig.text(
    1.1,
    -0.15,
    r"$t_{\rm univ}$ [Gyr]",
    ha="center",
    va="top",
    transform=ax[6].transAxes,
)
fig.text(1, 1.2, r"$z$", ha="left", va="top", transform=ax[0].transAxes)
for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)
# plt.savefig(
#     "./figures/cgm_temperature_and_density.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
plt.show()

 # %% similar to timescale ratios but get the contributions from the different dot E terms
import cmasher as cmr

# colormap
# cmap = matplotlib.cm.get_cmap("coolwarm")  # cmr.tropical_r
# cmap_segmented = matplotlib.cm.get_cmap("coolwarm", num_halos)
cmap = cmr.guppy_r
cmap_segmented = cmr.get_sub_cmap("cmr.guppy_r", 0, 1, N=num_halos)
# segmented the colormap based on the number of halo samples
num_halos = len(halo_masses)
colors = [cmap_segmented(i) for i in range(num_halos)]
norm = matplotlib.colors.LogNorm(
    vmin=halo_masses.value.min(), vmax=halo_masses.value.max()
)
sm = plt.cm.ScalarMappable(cmap=cmap_segmented, norm=norm)

fig, ax = plt.subplots(4, 1, figsize=(4.25, 8.5), dpi=300, sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.05, wspace=0.03)

threshold = 1e6  # K, example threshold for virial temperature

for res in halo_results:
    color = cmap(norm(res["mhalo"].value))
    times = res["sim_time_2"]
    redshift = cosmology.z_at_value(LCDM.age, times * u.Gyr).value
    dotE_tot = np.abs(
        res["dot_e_cgm_out"]
        + res["dot_e_cgm_in"]
        + res["dot_e_cgm_cooling"]
        + res["dot_e_ism_wind"]
    )
    y_eject = res["dot_e_cgm_out"] / dotE_tot
    y_cool = res["dot_e_cgm_cooling"] / dotE_tot
    y_in = res["dot_e_cgm_in"] / dotE_tot
    y_wind = res["dot_e_ism_wind"] / dotE_tot
    t_vir = res["t_vir"]
    t_cgm = res["cgm_temp"]

    mask = t_vir > threshold
    if np.sum(mask) > 0:
        begin = np.argmax(np.argwhere(t_vir < threshold).flatten())
        ax[0].plot(times[: begin + 2], y_cool[: begin + 2], ls="-", lw=2, color=color)
        ax[1].plot(times[: begin + 2], y_eject[: begin + 2], ls="-", lw=2, color=color)
        ax[2].plot(times[: begin + 2], y_in[: begin + 2], ls="-", lw=2, color=color)
        ax[3].plot(times[: begin + 2], y_wind[: begin + 2], ls="-", lw=2, color=color)

        ax[0].plot(times[mask], y_cool[mask], ls="--", lw=2, color=color)
        ax[1].plot(times[mask], y_eject[mask], ls="--", lw=2, color=color)
        ax[2].plot(times[mask], y_in[mask], ls="--", lw=2, color=color)
        ax[3].plot(times[mask], y_wind[mask], ls="--", lw=2, color=color)

    else:
        ax[0].plot(times, y_cool, ls="-", alpha=0.5, color=color)
        ax[1].plot(times, y_eject, ls="-", alpha=0.5, color=color)
        ax[2].plot(times, y_in, ls="-", alpha=0.5, color=color)
        ax[3].plot(times, y_wind, ls="-", alpha=0.5, color=color)

# for res in halo_results_baseline:
#     color = cmap(norm(res["mhalo"].value))
#     times = res["sim_time"]
#     redshift = cosmology.z_at_value(LCDM.age, times * u.Gyr).value

#     dotE_tot = np.abs(
#         res["dot_e_cgm_out"]
#         + res["dot_e_cgm_in"]
#         + res["dot_e_cgm_cooling"]
#         + res["dot_e_ism_wind"]
#     )
#     y_eject = res["dot_e_cgm_out"] / dotE_tot
#     y_cool = res["dot_e_cgm_cooling"] / dotE_tot
#     y_in = res["dot_e_cgm_in"] / dotE_tot
#     y_wind = res["dot_e_ism_wind"] / dotE_tot
#     t_vir = res["t_vir"]

#     mask = t_vir > threshold
#     if np.sum(mask) > 0:
#         begin = np.argmax(np.argwhere(t_vir < threshold).flatten())
#         ax[0, 1].plot(times[: begin + 2], y_cool[: begin + 2], ls="-", lw=2, color=color)
#         ax[1, 1].plot(times[: begin + 2], y_eject[: begin + 2], ls="-", lw=2, color=color)
#         ax[2, 1].plot(times[: begin + 2], y_in[: begin + 2], ls="-", lw=2, color=color)
#         ax[3, 1].plot(times[: begin + 2], y_wind[: begin + 2], ls="-", lw=2, color=color)

#         ax[0, 1].plot(times[mask], y_cool[mask], ls="--", lw=2, color=color)
#         ax[1, 1].plot(times[mask], y_eject[mask], ls="--", lw=2, color=color)
#         ax[2, 1].plot(times[mask], y_in[mask], ls="--", lw=2, color=color)
#         ax[3, 1].plot(times[mask], y_wind[mask], ls="--", lw=2, color=color)
#     else:
#         ax[0, 1].plot(times, y_cool, color=color, alpha=0.5)
#         ax[1, 1].plot(times, y_eject, color=color, alpha=0.5)
#         ax[2, 1].plot(times, y_in, color=color, alpha=0.5)
#         ax[3, 1].plot(times, y_wind, color=color, alpha=0.5)

ax[0].set(
    yscale="log",
    ylabel=r"$\dot{E}_{\rm CGM, cool}/\dot{E}_{\rm CGM, added}$",
    xlim=(0.08, 5),
    ylim=(2e-3, 1),
    xscale="log",
)
ax[1].set(yscale="log")
ax[1].set(
    yscale="log",
    ylabel=r"$\dot{E}_{\rm CGM,ej}/\dot{E}_{\rm CGM, added}$",
)
ax[1].set(yscale="log")
ax[2].set(
    yscale="log",
    ylabel=r"$\dot{E}_{\rm CGM,acc}/\dot{E}_{\rm CGM, added}$",
)
ax[2].set(yscale="log")
ax[3].set(
    yscale="log",
    ylabel=r"$\dot{E}_{\rm SNe,wind}/\dot{E}_{\rm CGM, added}$",
    xlabel=r"$t_{\rm univ} [\mathrm{Gyr}]$",
)
ax[3].set(
    yscale="log",
    xlabel=r"$t_{\rm univ} [\mathrm{Gyr}]$",
)

# add colorbar for halo mass (log scale)
cbar_ax = ax[1].inset_axes([0.3, 0.35, 0.6, 0.08])  # [x, y, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label(r"$M_{\rm halo} (z = 0) \ [{\rm M_\odot}]$")
cbar.set_ticks(halo_masses.value)
cbar.ax.set_xscale("log")

# add twin redshift axis for the top row of plots (indices 0 and 1)
for i in [0]:
    t_ticks = np.array([0.14, 0.3, 0.5, 1, 2, 3, 5])
    z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
    ax2 = ax[0].twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax[i].get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    ax2.set_xlabel(r"$z$", labelpad=8)
    ax2.minorticks_off()
    ax[i].minorticks_on()

# # add a reference line for y = 1
# for row in range(4):
#     for col in range(2):
#         ax[row, col].axhline(1, ls="-", lw=2, color="grey", alpha=0.5)

# for i in range(4):
#     ax[i].axhline(0.5, ls="-", lw=2, color="grey", alpha=0.5)
# add text
ax[0].text(
    0.90,
    0.15,
    "this work",
    transform=ax[0].transAxes,
    fontsize=12,
    ha="right",
    va="top",
)
# ax[0, 1].text(
#     0.10,
#     0.15,
#     "baseline C23 model",
#     transform=ax[0, 1].transAxes,
#     fontsize=12,
#     ha="left",
#     va="top",
# )


for axes in ax.flatten():
    for line in axes.lines:
        line.set_zorder(1)

# plt.savefig(
#     "./figures/dotE_contribution_ratios.png",
#     dpi=200,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )

plt.show()
# %%
fig, ax = plt.subplots(1, 1, dpi=300, figsize=(4.5, 4), sharex="col", sharey="row")

for res in halo_results[:1]:
    color = cmap(norm(res["mhalo"].value))
    times = res["sim_time_2"]
    t_vir = res["t_vir"]
    t_cgm = res["cgm_temp"]
    rho0 = res["rho0"]
    dot_e_plus = res["dot_e_plus"]
    dot_e_minus = res["dot_e_minus"]
    dotE = dot_e_plus - dot_e_minus
    ax.plot(
        times,
        dot_e_plus,
        color=color,
        label=r"$\dot{E}_{\rm CGM,in} + \dot{E}_{\rm SNe, wind}$",
    )
    ax.plot(
        times,
        dot_e_minus,
        color=color,
        ls="--",
        label=r"$\dot{E}_{\rm CGM,cool} + \dot{E}_{\rm CGM, ej}$",
    )
ax.legend(frameon=False, fontsize=10)
ax.set(
    yscale="log",
    xlim=(0.05, 5),
    ylabel=r"$\rho_0 ~ [{\rm M_\odot ~kpc^{-3}}]$",
    ylim=(1e52, 1e56),
)
