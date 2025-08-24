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

# %% Plot timescales for baseline and latest (2-phase) models in a single figure

mhalo_z0 = 1e12 * u.Msun
t_span = (0.5, 13)  # gyrs

# Baseline model
baseline_model = CGMRegulatorBaseline(mhalo_z0, t_span)
baseline_model.run_halo()
results_baseline = baseline_model.get_results()
derived_baseline = baseline_model.get_derived_quantities()

# 2-phase (latest) model
model_2phase = CGMRegulator(mhalo_z0, t_span)
model_2phase.run_halo()
results_2phase = model_2phase.get_results()
derived_2phase = model_2phase.get_derived_quantities()

cmap = plt.get_cmap("Dark2")
color_star = cmap(5)
color_green = cmap(4)
color_cgm = cmap(2)
color_baseline = cmap(0)
color_latest = cmap(1)
#%%
fig, axes = plt.subplots(2, 1, figsize=(4.5, 6), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.1)

# --- Top panel: Baseline model timescales ---
ax = axes[0]
t_cool = derived_baseline["tcool_real"]
t_dynamical = derived_baseline["t_dyn"]
t_cool_eff = derived_baseline["tcool_eff"]
t_dep = derived_baseline["t_dep"]
t_ejection = derived_baseline["t_ejection"]
sim_time = derived_baseline["sim_time"]

ax.plot(sim_time, t_cool, lw=3, color=color_cgm, label=r"$t_{\mathrm{cool}}$")
ax.plot(sim_time, t_dynamical, lw=3, color="grey", label=r"$t_{\mathrm{ff}}$")
ax.plot(sim_time, t_cool_eff, lw=3, color=color_latest, label=r"$t_{\mathrm{cool,eff}}$")
ax.plot(sim_time, t_ejection, lw=3, color=color_green, label=r"$t_{\mathrm{ejection}}$", ls="--")
ax.plot(sim_time, t_dep, lw=3, color=color_star, label=r"$t_{\mathrm{dep}}$")

ax.set(
    yscale="log",
    xscale="log",
    ylim=(1e-3, 12),
    xlim=(results_baseline["t"][0] * 0.8, 13),
    ylabel=r"timescales $[\mathrm{Gyr}]$"
)
ax.legend(frameon=False, ncol=2, fontsize=10, title="baseline ")


# Twin redshift axis for top panel
t_ticks = np.array([0.3, 0.5, 1, 2, 4, 8, results_baseline["t"][-1]])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
ax2_top = ax.twiny()
ax2_top.plot(results_baseline["t"], results_baseline["m_star"], color="k", alpha=0)
ax2_top.set(xscale="log", xlim=(0.25, results_baseline["t"][-1]),)
ax2_top.set_xticks(t_ticks)
ax2_top.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
ax2_top.set_xlabel(r"$z$", labelpad=8)
ax2_top.minorticks_off()
ax.minorticks_on()
# --- Bottom panel: 2-phase (latest) model timescales ---
ax = axes[1]
t_cool_2 = derived_2phase["tcool_real"]
t_dynamical_2 = derived_2phase["t_dynamical"]
t_dep_effect = results_2phase["m_star"] / derived_2phase["dot_m_sfr"]
t_ejection_2 = derived_2phase["t_ejection"]
sim_time_2 = derived_2phase["sim_time"]

ax.plot(sim_time_2, t_cool_2, lw=3, color=color_cgm, label=r"$t_{\mathrm{cool}}$")
ax.plot(sim_time_2, t_dynamical_2, lw=3, color="grey", label=r"$t_{\mathrm{ff}}$")
ax.plot(sim_time_2, t_dep_effect, lw=3, color=color_latest, label=r"$t_{\mathrm{dep, eff}}$")
ax.plot(sim_time_2, t_ejection_2, lw=3, color=color_green, label=r"$t_{\mathrm{ejection}}$", ls="--")
# Overlay baseline depletion time for comparison
ax.plot(sim_time, t_dep, lw=3, color=color_star, label=r"$t_{\mathrm{dep}}$", ls="--")

ax.set(
    yscale="log",
    xscale="log",
    ylim=(1e-3, 12),
    xlim=(results_2phase["t"][0] * 0.8, 13),
    ylabel=r"timescales $[\mathrm{Gyr}]$"
)
ax.legend(frameon=False, ncol=2, fontsize=10, title="2 phase")

ax.set_xlabel(r"time $[\mathrm{Gyr}]$")

# Share x-axis between panels
axes[1].set_xlim(axes[0].get_xlim())

plt.savefig(
    "./figures/model_timescales_comparison.png", dpi=200, bbox_inches="tight", pad_inches=0.05
)
plt.show()

# %%
