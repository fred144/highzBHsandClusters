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

# %% do a single run of each model
mhalo_z0 = 1e12 * u.Msun
t_span = (0.5, 13)  # gyrs

baseline_model = CGMRegulatorBaseline(
    mhalo_z0,
    t_span,
)
run = baseline_model.run_halo()
results = baseline_model.get_results()
derived = baseline_model.get_derived_quantities()

print(results.keys())
print(derived.keys())

fig, ax = plt.subplots(1, 2, figsize=(4.8, 2), dpi=300, sharex=True)
plt.subplots_adjust(wspace=0.42)
cmap = plt.get_cmap("Dark2")
# Color assignments for clarity
color_star = cmap(5)  # Stellar mass
color_green = cmap(4)  # ISM mass
color_cgm = cmap(2)  # CGM mass
color_baseline = cmap(0)  # Baseline model
color_latest = cmap(1)  # Latest model

ax[0].plot(derived["sim_time"], derived["dot_m_sfr"], lw=3, color=color_baseline)
ax[0].set(
    ylabel=r"$\dot{M}_{\star}$ [M$_{\odot}$ Gyr$^{-1}$]",
    yscale="log",
    ylim=(2e4, 8e9),
    xlabel=r"time $[\mathrm{Gyr}]$",
)

ax[0].text(
    0.95,
    0.05,
    r"$M_{{\rm halo}} (z=0) = 10^{{12}}$",
    transform=ax[0].transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
)

# now, SFE as a function of time
ax[1].plot(derived["sim_time"], derived["f_star"], lw=3, color=color_baseline)
ax[1].set(
    ylabel=r"$\mathrm{SFE} = M_{\star} / (f_{\rm b} M_{\rm halo})$",
    yscale="log",
    ylim=(5e-4, 1),
    xlabel=r"time $[\mathrm{Gyr}]$",
)

# add an inset for the timescales
ax_inset = ax[0].inset_axes([0.0, 1.35, 2.4, 1])
# now get timescales
t_cool = derived["tcool_real"]  # cooling
t_dynamical = derived["t_dyn"]  # dynamical
t_cool_eff = derived["tcool_eff"]  # effective cooling, t_cool+t_dynamical
t_dep = derived["t_dep"]  # depletion time, denominator for SFR
t_dep_baseline = t_dep
t_baseline = derived["sim_time"]
t_ejection = derived["t_ejection"]
# plot these as a function of time
ax_inset.plot(
    derived["sim_time"], t_cool, lw=3, color=color_cgm, label=r"$t_{\mathrm{cool}}$"
)
ax_inset.plot(
    derived["sim_time"], t_dynamical, lw=3, color="grey", label=r"$t_{\mathrm{ff}}$"
)
ax_inset.plot(
    derived["sim_time"],
    t_cool_eff,
    lw=3,
    color=color_latest,
    label=r"$t_{\mathrm{cool,eff}}$",
)
ax_inset.plot(
    derived["sim_time"], t_ejection, lw=3, color=color_green, label=r"$t_{\mathrm{ejection}}$", ls="--"
)

ax_inset.set(
    ylabel=r"timescales $[\mathrm{Gyr}]$",
    yscale="log",
    xscale="log",
    ylim=(1e-3, 12),
    xlim=(results["t"][0] * 0.8, 13),
    xlabel=r"time $[\mathrm{Gyr}]$",
)
ax_inset.plot(
    derived["sim_time"], t_dep, lw=3, color=color_star, label=r"$t_{\mathrm{dep}}$"
)
ax_inset.legend(frameon=False, ncol=2)

# make a twin redshift axis for the top row, using z
# get the current x axis labels of the first row and their
# Get more ticks for the twin axis by interpolating between min and max time
t_ticks = np.array([0.3, 0.5, 1, 2, 4, 8, results["t"][-1]])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]

for i, a in enumerate(ax):
    if (i == 0) or (i == 2):
        continue
    ax2 = ax_inset.twiny()
    # dummy plot to sync axis
    ax2.plot(results["t"], results["m_star"], color="k", alpha=0)
    ax2.set(xscale="log", xlim=(0.25, results["t"][-1]))
    if i == 1:
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
        ax2.set_xlabel(r"$z$")
    else:
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels([])

    ax2.minorticks_off()
ax_inset.minorticks_on()
for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)

plt.savefig(
    "./figures/baseline_model_sfe.png", dpi=200, bbox_inches="tight", pad_inches=0.05
)

plt.show()

# %%

model_2phase = CGMRegulator(mhalo_z0, t_span)
run = model_2phase.run_halo()
results = model_2phase.get_results()
derived = model_2phase.get_derived_quantities()

fig, ax = plt.subplots(1, 2, figsize=(4.8, 2), dpi=300, sharex=True)
plt.subplots_adjust(wspace=0.42)


ax[0].plot(derived["sim_time"], derived["dot_m_sfr"], lw=3, color=color_baseline)
ax[0].set(
    ylabel=r"$\dot{M}_{\star}$ [M$_{\odot}$ Gyr$^{-1}$]",
    yscale="log",
    ylim=(2e4, 8e9),
    xlabel=r"time $[\mathrm{Gyr}]$",
)

ax[0].text(
    0.95,
    0.05,
    r"$M_{{\rm halo}} (z=0) = 10^{{12}}$",
    transform=ax[0].transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
)

# now, SFE as a function of time
ax[1].plot(derived["sim_time"], derived["halo_sfe"], lw=3, color=color_baseline)
ax[1].set(
    ylabel=r"$\mathrm{SFE} = M_{\star} / (f_{\rm b} M_{\rm halo})$",
    yscale="log",
    ylim=(5e-4, 1),
    xlabel=r"time $[\mathrm{Gyr}]$",
)

# add an inset for the timescales
ax_inset = ax[0].inset_axes([0.0, 1.35, 2.4, 1])
# now get timescales
t_cool = derived["tcool_real"]  # cooling
t_dynamical = derived["t_dynamical"]  # dynamical
t_dep_effect = (
    results["m_star"] / derived["dot_m_sfr"]
)  # depletion time, denominator for SFR
t_ejection = derived["t_ejection"]
# plot these as a function of time
ax_inset.plot(
    derived["sim_time"], t_cool, lw=3, color=color_cgm, label=r"$t_{\mathrm{cool}}$"
)
ax_inset.plot(
    derived["sim_time"], t_dynamical, lw=3, color="grey", label=r"$t_{\mathrm{ff}}$"
)
ax_inset.plot(
    derived["sim_time"],
    t_dep_effect,
    lw=3,
    color=color_latest,
    label=r" $t_{\mathrm{dep, eff}}$",
)
ax_inset.plot(
    derived["sim_time"], t_ejection, lw=3, color=color_green, label=r"$t_{\mathrm{ejection}}$", ls="--"
)

ax_inset.set(
    ylabel=r"timescales $[\mathrm{Gyr}]$",
    yscale="log",
    xscale="log",
    ylim=(1e-3, 12),
    xlim=(results["t"][0] * 0.8, 13),
    xlabel=r"time $[\mathrm{Gyr}]$",
)
ax_inset.plot(
    t_baseline,
    t_dep_baseline,
    lw=3,
    color=color_star,
    label=r"$t_{\mathrm{dep}}$",
    ls="--",
)
ax_inset.legend(frameon=False, ncol=2)

# make a twin redshift axis for the top row, using z
# get the current x axis labels of the first row and their
# Get more ticks for the twin axis by interpolating between min and max time
t_ticks = np.array([0.3, 0.5, 1, 2, 4, 8, results["t"][-1]])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]

for i, a in enumerate(ax):
    if (i == 0) or (i == 2):
        continue
    ax2 = ax_inset.twiny()
    # dummy plot to sync axis
    ax2.plot(results["t"], results["m_star"], color="k", alpha=0)
    ax2.set(xscale="log", xlim=(0.25, results["t"][-1]))
    if i == 1:
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
        ax2.set_xlabel(r"$z$")
    else:
        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels([])

    ax2.minorticks_off()
ax_inset.minorticks_on()
for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)

plt.savefig(
    "./figures/2phase_model_sfe.png", dpi=200, bbox_inches="tight", pad_inches=0.05
)

plt.show()

# %%
