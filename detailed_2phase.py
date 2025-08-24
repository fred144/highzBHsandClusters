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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from run_grids_of_models import (
    run_baseline_model_redshift_grid,
    run_2phase_model_redshift_grid,
)
import matplotlib.lines as mlines
from matplotlib.patches import ConnectionPatch

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
t_span = (0.05, 13.3)  # gyrs

baseline_model = CGMRegulatorBaseline(
    mhalo_z0,
    t_span,
)
run = baseline_model.run_halo()
results = baseline_model.get_results()
derived = baseline_model.get_derived_quantities()
# %%
model_2phase = CGMRegulator(
    mhalo_z0,
    t_span,
    tstep=0.005,
)
run_2phase = model_2phase.run_halo()
results_2phase = model_2phase.get_results()
derived_2phase = model_2phase.get_derived_quantities()
# %% take a detailed look at the change in the CGM mass and energy

tmin = results_2phase["t"].min() * 0.1
tmax = results_2phase["t"].max()

print(derived_2phase.keys())
print(results_2phase.keys())

fig, ax = plt.subplots(
    4, 2, figsize=(10, 9.0), dpi=300, gridspec_kw={"height_ratios": [2.5, 0.5, 3, 3]}
)
plt.subplots_adjust(hspace=0.05, wspace=0.2)
ax = ax.flatten()

cmap = plt.get_cmap("Dark2")
hot_clr = cmap(1)
col_clr = cmap(0)

cmapdark2 = plt.get_cmap("Dark2")
total_clr = cmapdark2(0)

# Panel 0: CGM mass rates

ax[0].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_hot"],
    color=hot_clr,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, hot}$",
)
ax[0].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_cold"],
    color=col_clr,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, cold}$",
)

ax[0].legend(frameon=False, ncol=1)
ax[0].set(
    ylabel=r"$\dot{M}_{\rm CGM}  ~ [{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
    yscale="log",
    ylim=(2e5, 2e10),
)

# Panel 3: CGM mass rates (inset range)
ax[4].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_hot"],
    color=hot_clr,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, hot}$",
)
ax[4].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_cold"],
    color=col_clr,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, cold}$",
)
ax[4].set(yscale="log", xlim=(0.05, 1.8), ylim=(4e5, 8e9))


# Panel 1: CGM energy rates
dot_e_sne_wind = derived_2phase["dot_e_ism_wind"]
dot_e_cgm_acc = derived_2phase["dot_e_cgm_in"]
dot_e_cgm_cool = derived_2phase["dot_e_cgm_cooling"]
dot_e_cgm_ej = derived_2phase["dot_e_cgm_out"]
dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool

ax[1].plot(
    results_2phase["t"],
    dot_e_cgm_total,
    color=hot_clr,
    lw=2,
    # label=r"$\dot{E}_{\rm CGM, total}$",
)
ax[1].set(ylabel=r"$\dot{E}_{\rm CGM, total}~ [{\rm erg\: Gyr^{-1}}]$", yscale="log", ylim=(2e52, 2e58))
ax[1].legend(frameon=False, ncol=1, loc="lower right")

# Panel 4: CGM energy rates (inset range)
ax[5].plot(
    results_2phase["t"],
    dot_e_cgm_total,
    color=hot_clr,
    lw=2,
    label=r"$\dot{E}_{\rm CGM, total}$",
)
ax[5].set(yscale="log", xlim=(0.05, 1.8), ylim=(5e52, 6e57))


# --- Custom connectors ---
# Connector 1: top-left of zoom region in parent → top-left corner of inset
con1 = ConnectionPatch(
    xyA=(0, 1),
    coordsA=ax[5].transAxes,
    xyB=(0.05, 5e52),
    coordsB=ax[1].transData,
    color="k",
    lw=1,
    zorder=10,
)

# Connector 2: bottom-right of zoom region in parent → bottom-right corner of inset
con2 = ConnectionPatch(
    xyA=(1, 1),
    coordsA=ax[5].transAxes,
    xyB=(1.8, 5e52),
    coordsB=ax[1].transData,
    color="k",
    lw=1,
    zorder=10,
)
ax[1].figure.add_artist(con1)
ax[1].figure.add_artist(con2)
artists = mark_inset(
    ax[1],
    ax[5],
    loc1=2,
    loc2=4,
    fc="none",
    color="k",
    zorder=10,
)
for line in artists[1:]:
    line.set_alpha(0.0)


# Connector 1: top-left of zoom region in parent → top-left corner of inset
con1 = ConnectionPatch(
    xyA=(0, 1),
    coordsA=ax[4].transAxes,  # UL corner of inset
    xyB=(0.05, 4e5),
    coordsB=ax[0].transData,  # UL of parent zoom
    color="k",
    lw=1,
    zorder=10,
)
ax[0].figure.add_artist(con1)

# Connector 2: bottom-right of zoom region in parent → bottom-right corner of inset
con2 = ConnectionPatch(
    xyA=(1, 1),
    coordsA=ax[4].transAxes,  # LR corner of inset
    xyB=(1.8, 4e5),
    coordsB=ax[0].transData,  # LR of parent zoom
    color="k",
    lw=1,
    zorder=10,
)
ax[0].figure.add_artist(con2)

# --- Draw rectangle only, hide default connectors ---
artists = mark_inset(ax[0], ax[4], loc1=2, loc2=4, fc="none", color="k", zorder=10)
for line in artists[1:]:
    line.set_alpha(0.0)

# Panels 2 and 3: ghost rows for spacing only, no plots
ax[2].axis("off")
ax[3].axis("off")

# Panel 4: CGM mass rates (zoomed)
dot_m_cgm_cooling = results_2phase["m_cgm_hot"] / derived_2phase["tcool_real"]
dot_m_cgm_in = derived_2phase["dot_m_cgm_in"]
dot_m_sne_wind = derived_2phase["dot_m_ism_wind"]
dot_m_cgm_ej = derived_2phase["dot_m_cgm_out"]
dot_cgm_falling = results_2phase["m_cgm_cold"] / derived_2phase["t_dynamical"]

red1 = "tab:red"
red2 = "tab:orange"
blu1 = "dodgerblue"
blu2 = "tab:green"



ax[6].plot(
    results_2phase["t"],
    dot_m_cgm_ej,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, ej}$",
    color=blu2,
)


ax[6].plot(
    results_2phase["t"],
    dot_cgm_falling,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, falling}$",
    color="y",
)
ax[6].plot(
    results_2phase["t"],
    dot_m_cgm_cooling,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, cooling}$",
    color=blu1,
)
ax[6].plot(
    results_2phase["t"],
    dot_m_cgm_in,
     ls="--",
    lw=2,
    label=r"$\dot{M}_{\rm CGM, acc}$",
    color=red1,
)
ax[6].plot(
    results_2phase["t"],
    dot_m_sne_wind,
    ls="--",
    lw=2,
    label=r"$\dot{M}_{\rm SNe, wind}$",
    color=red2,
)
ax[6].set(
    ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
    yscale="log",
    ylim=(3e5, 2e10),
    xlabel=r"time $[\mathrm{Gyr}]$",
    xlim=(0.05, 1.8),
)
ax[6].legend(frameon=False, ncol=2, fontsize=10)

ax[6].text(
    0.42,
    0.42,
    r"removes hot gas",
    transform=ax[6].transAxes,
    fontsize=10,
    ha="center",
    va="top",
)
ax[6].text(
    0.8,
    0.42,
    r"adds hot gas",
    transform=ax[6].transAxes,
    fontsize=10,
    ha="center",
    va="top",
)
ax[6].text(
    0.8,
    0.12,
    r"$\leftarrow$ adds to cold gas",
    transform=ax[6].transAxes,
    fontsize=10,
    ha="center",
    va="top",
)

# Panel 5: CGM energy rates (zoomed)
ax[7].plot(
    results_2phase["t"],
    dot_e_cgm_acc,
    color=red1,
     ls="--",
    lw=2,
    label=r"$\dot{E}_{\rm CGM, acc}$",
)
ax[7].plot(
    results_2phase["t"],
    dot_e_sne_wind,
     ls="--",
    color=red2,
    lw=2,
    label=r"$\dot{E}_{\rm SNe, wind}$",
)
ax[7].plot(
    results_2phase["t"],
    dot_e_cgm_ej,
    color=blu2,
    lw=2,
    label=r"$\dot{E}_{\rm CGM, ej}$",
)
ax[7].plot(
    results_2phase["t"],
    dot_e_cgm_cool,
    color=blu1,
    lw=2,
    label=r"$\dot{E}_{\rm CGM, cooling}$",
)
ax[7].set(
    ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
    yscale="log",
    ylim=(5e52, 9.5e57),
    xlim=(0.05, 1.8),
    xlabel=r"time $[\mathrm{Gyr}]$",
)
ax[7].legend(frameon=False, ncol=2, fontsize=10)

# first column of the legend add energy to the CGM , 2nd removes, add text annotation
ax[7].text(
    0.42,
    0.32,
    r"adds energy",
    transform=ax[7].transAxes,
    fontsize=10,
    ha="center",
    va="top",
)
ax[7].text(
    0.8,
    0.32,
    r"removes energy",
    transform=ax[7].transAxes,
    fontsize=10,
    ha="center",
    va="top",
)

# make a twin redshift axis for the top row, using z
t_ticks = np.array([1, 5, 8, 10, 12, 13.3])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]

for i in [0, 1]:
    ax2 = ax[i].twiny()
    ax2.plot(results_2phase["t"], derived_2phase["dot_m_cgm_hot"], color="k", alpha=0)
    ax2.set_xlim((None, tmax))
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    ax2.set_xlabel(r"$z$")
    ax2.minorticks_off()
    ax[i].minorticks_on()
    ax[i].set_xlim((None, tmax))

t_ticks = np.array([0.3, 0.6, 1, 1.5])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
for i in [4, 5]:
    ax2 = ax[i].twiny()
    ax2.plot(results_2phase["t"], derived_2phase["dot_m_cgm_hot"], color="k", alpha=0)
    ax2.set_xlim((0.05, 1.8))
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    # ax2.set_xlabel(r"$z$")
    ax2.minorticks_off()
    ax[i].minorticks_on()
    ax[i].set_xlim((0.05, 1.8))
    # turn x axis ticks off
    ax[i].set_xticks([])

plt.savefig(
    "./figures/2phase_detailed_cgm_1e12.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.show()


# %%
