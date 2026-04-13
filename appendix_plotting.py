# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from astropy import cosmology
import scipy
import cmasher as cmr
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator

# import seaborn as sns
from regulator_lib.cooling_fn_generator import cooling_fn_generator
import astropy.constants as consts
import astropy.units as u
import warnings
from cgm_sf_regulator import CGMRegulator
from tqdm import tqdm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import time
from matplotlib.colors import SymLogNorm
from regulator_lib.cooling_fn_generator import CoolingFunctionInterpolator


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

# standard flat cosmology
H0 = 70
h = 0.7
Omegam0 = 0.3
Omegade0 = 0.7
Ob0 = 0.0490
f_baryon = Ob0 / Omegam0  # universal baryon fraction
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)

t_span = (0.15, 1.0)  # Gyr
f_prevent_floor_to_try = [0.01,  0.2, 0.5, 0.6, 1.0]
mhalo_0 = 1e12 * u.Msun
z0_text = r"$M_{\rm halo}(z=0) = 10^{12} ~ {\rm M_\odot}$"

"""
f_prevent analysis
"""

results_list = []
for idx, f_prevent_floor in enumerate(f_prevent_floor_to_try):
    model_ = CGMRegulator(
        mhalo_0,
        t_span,
        tstep=0.001,
        add_f_prevent_floor=f_prevent_floor,
        verbose=False,
        updated_loadings=True,
        updated_halo_infall=True,
    )
    run_ = model_.run_halo()
    results_ = model_.get_results()
    derived_ = model_.get_derived_quantities()
    dot_e_cgm_actual = (
        derived_["dot_e_cgm_in"]
        - derived_["dot_e_cgm_cooling"]
        - derived_["dot_e_cgm_out"]
        + derived_["dot_e_ism_wind"]
    )
    dot_m_cgm_actual =derived_["dot_m_cgm_hot"]
      
        
   
   
    t_dynamical = derived_["t_dynamical"]
    t_cool = derived_["tcool_real"]
    t_dep_effect = results_["m_star"] / derived_["dot_m_sfr"]
    t_ejection = derived_["t_ejection"]
    sim_time = derived_["sim_time"]
    results_list.append(
        {
            "idx": idx,
            "label": f"{f_prevent_floor}",
            "time_gyr": results_["t"],
            "mstar": results_["m_star"],
            "m_cgm_cold": results_["m_cgm_cold"],
            "m_cgm_hot": results_["m_cgm_hot"],
            "dot_m_cgm_ej": derived_["dot_m_cgm_out"],
            "dot_cgm_falling": results_["m_cgm_cold"] / derived_["t_dynamical"],
            "dot_m_cgm_in": derived_["dot_m_cgm_in"],
            "dot_m_sne_wind": derived_["dot_m_ism_wind"],
            "dot_m_cgm_cooling": results_["m_cgm_hot"] / derived_["tcool_real"],
            "e_cgm": results_["egy_cgm"],
            "dot_e_cgm_out": derived_["dot_e_cgm_out"],
            "dot_e_cgm_cooling": derived_["dot_e_cgm_cooling"],
            "dot_e_cgm_in": derived_["dot_e_cgm_in"],
            "dot_e_ism_wind": derived_["dot_e_ism_wind"],
            "dot_e_cgm_actual": dot_e_cgm_actual,
            "t_dynamical": t_dynamical,
            "t_cool": t_cool,
            "t_dep": t_dep_effect,
            "t_ejection": t_ejection,
            "T_CGM": derived_["cgm_temp"],
            "T_vir": derived_["halo_vir_temp"],
            "dot_m_cgm_actual": dot_m_cgm_actual,
        }
    )
# %% # ===== PLOTTING =====
# Create colormap with first color grey and rest from viridis
viridis_colors = plt.cm.Dark2_r(np.linspace(0, 1, len(f_prevent_floor_to_try)))
colors = viridis_colors  # np.vstack([[0, 0, 0, 1.0], viridis_colors])

fig, ax = plt.subplots(
    2,
    4,
    figsize=(12, 6),
    dpi=300,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1]},
)

plt.subplots_adjust(hspace=0.05, wspace=0.3)

lw = 2
# add y grid
for axes in ax.flatten():
    axes.grid(which="both", ls="-", lw=0.5, alpha=0.4, zorder=0)

for i, res in enumerate(results_list):
    ls = "-" if i % 2 == 0 else "--"
    ax[0, 0].plot(
        res["time_gyr"],
        res["mstar"],
        color=colors[i],
        label=res["label"],
        lw=lw,
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[0, 1].plot(
        res["time_gyr"],
        res["m_cgm_cold"],
        color=colors[i],
        lw=lw,
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[0, 2].plot(
        res["time_gyr"],
        res["m_cgm_hot"],
        color=colors[i],
        lw=lw,
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[0, 3].plot(
        res["time_gyr"],
        res["dot_m_cgm_ej"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )

    ax[1, 0].plot(
        res["time_gyr"],
        res["dot_cgm_falling"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[1, 1].plot(
        res["time_gyr"],
        res["dot_m_cgm_in"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[1, 2].plot(
        res["time_gyr"],
        res["dot_m_sne_wind"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[1, 3].plot(
        res["time_gyr"],
        res["dot_m_cgm_cooling"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )

ax[0, 0].set(yscale="log", ylabel=r"Stellar Mass [M$_{\odot}$]", ylim=(1e5, 1e8))
ax[0, 1].set(ylabel=r"CGM Cold Mass [M$_{\odot}$]", yscale="log", ylim=(1e5, 1e8))
ax[0, 2].set(ylabel=r"CGM Hot Mass [M$_{\odot}$]", yscale="log", ylim=(1e5, 1e8))
ax[0, 3].set(
    ylabel=r"$\dot{M}_{\rm CGM, ej} ~ {[ \rm M_\odot ~ yr^{-1}] }$",
    yscale="log",
    ylim=(3e5, 7e9),
)

ax[1, 0].set(
    ylabel=r"$\dot{M}_{\rm CGM, falling}~ {[ \rm M_\odot ~ yr^{-1}] }$",
    yscale="log",
    ylim=(3e5, 7e9),
    xlabel=r"$t_{\rm univ}$ [Gyr]",
)
ax[1, 1].set(
    ylabel=r"$\dot{M}_{\rm CGM, in}~ {[ \rm M_\odot ~ yr^{-1}] }$",
    yscale="log",
    ylim=(3e5, 7e9),
    xlabel=r"$t_{\rm univ}$ [Gyr]",
)
ax[1, 2].set(
    ylabel=r"$\dot{M}_{\rm SNe, wind}~ {[ \rm M_\odot ~ yr^{-1}] }$",
    yscale="log",
    ylim=(3e5, 7e9),
    xlabel=r"$t_{\rm univ}$ [Gyr]",
)
ax[1, 3].set(
    ylabel=r"$\dot{M}_{\rm CGM, cooling}~ {[ \rm M_\odot ~ yr^{-1}] }$",
    yscale="log",
    ylim=(3e5, 7e9),
    xlim=t_span,
    xlabel=r"$t_{\rm univ}$ [Gyr]",
)

t_ticks = np.array([t_span[0], 0.22, 0.3, 0.4, t_span[1]])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
for col in range(4):
    ax2 = ax[0, col].twiny()
    ax2.set_xlim(t_span)
    ax2.set_xlim(ax[0, col].get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    ax2.set_xlabel(r"$z$", labelpad=8)
    ax2.minorticks_off()
    ax[0, col].minorticks_on()

for axes in ax.flatten():
    for line in axes.lines:
        line.set_zorder(1)

ax[0, 0].legend(
    frameon=False,
    ncols=8,
    fontsize=9,
    title=r"$f_{\mathrm{prevent}} \rm ~floor$",
    loc="lower left",
    bbox_to_anchor=(0.75, 1.15),
    title_fontsize=10,
)

# add anotation for z0 halo mass
ax[0, 0].text(
    0.95,
    0.05,
    z0_text,
    transform=ax[0, 0].transAxes,
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    ha="right",
    va="bottom",
)

plt.savefig(
    "./final_figs/appendix_mass_fprevent_Mhalo{:.2f}.pdf".format(mhalo_0.value), dpi=200, bbox_inches="tight", pad_inches=0.05
)
plt.show()
# %% now do the same for the energy rates, total energy, total energy rate, then breakdown of energy rates

fig, ax = plt.subplots(
    3,
    3,
    figsize=(12, 8),
    dpi=300,
    sharex=True,
    gridspec_kw={"height_ratios": [ 1, 1,1]},
)

plt.subplots_adjust(hspace=0.05, wspace=0.3)
lw = 2
# add y grid
for axes in ax.flatten():
    axes.grid(which="both", ls="-", lw=0.5, alpha=0.4, zorder=0)    
    
for i, res in enumerate(results_list):
    # ls = "-" if i % 2 == 0 else "--"
    ax[0, 0].plot(
        res["time_gyr"],
        # res["e_cgm"],
        res["T_CGM"],
        color=colors[i],
     
        lw=lw,
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    
    ax[0, 1].plot(
        res["time_gyr"],
        res["dot_e_cgm_actual"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
        label=res["label"],
    )
    ax[0, 2].plot(
        res["time_gyr"],
        res["dot_e_cgm_out"]/res["dot_m_cgm_ej"],
        color=colors[i],
        lw=lw,
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[1, 0].plot(
        res["time_gyr"],
        res["dot_e_cgm_cooling"]/res["dot_m_cgm_cooling"],
        color=colors[i],
        lw=lw,
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[1, 1].plot(
        res["time_gyr"],
        res["dot_e_cgm_in"]/res["dot_m_cgm_in"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )
    ax[1, 2].plot(
        res["time_gyr"],
        res["dot_e_ism_wind"]/res["dot_m_sne_wind"],
        lw=lw,
        color=colors[i],
        alpha=0.7,
        linestyle=ls,
        zorder=2,
    )   
    ax[2, 0].plot(res["time_gyr"], res["t_cool"], lw=lw, color=colors[i], alpha=0.7, linestyle=ls, zorder=2)
    
    # ax[2, 1].plot(res["time_gyr"], res["dot_e_cgm_in"], lw=lw, color=colors[i], alpha=0.7, linestyle=ls, zorder=2)
    ax[2, 1].plot(res["time_gyr"], res["dot_m_cgm_in"], lw=lw, color=colors[i], alpha=0.7, linestyle=ls, zorder=2)
    
    
    
    
    # ax[2, 1].plot(res["time_gyr"], res["t_dep"], lw=lw, color=colors[i], alpha=0.7, linestyle=ls, zorder=2)
    ax[2, 2].plot(res["time_gyr"],  res["t_dep"],  lw=lw, color=colors[i], alpha=0.7, linestyle=ls, zorder=2)
    
# add the free fall time / dynamical time to the timescale plots

ax[2, 0].plot(res["time_gyr"], res["t_dynamical"], lw=3, color="k", alpha=1, linestyle=":", zorder=2, label=r"$t_{\rm ff}$")
ax[2, 2].plot(res["time_gyr"], res["t_dynamical"], lw=3, color="k", alpha=1, linestyle=":", zorder=2, label=r"$t_{\rm ff}$")

ax[2, 0].legend(frameon=False, fontsize=11, loc="lower right")

    
ax[0, 0].set(yscale="log", ylabel=r"$T_{\rm CGM}$ [K]",ylim=(3e4, 1e7))
# overplot th Tvir
ax[0, 0].plot(res["time_gyr"], res["T_vir"], lw=3, color="k", alpha=1, linestyle="--", zorder=2, label=r"$T_{\rm vir}$")
ax[0, 0].legend(frameon=False, fontsize=11, loc="lower right")
ax[0, 1].set(
    ylabel=r"$\dot{E}_{\rm CGM}$" + r" [erg Gyr$^{-1}$]",
    yscale="log",
    # ylim=(5e45, 3e48),
)
ax[0, 2].set(
    ylabel=r"$\dot{E}_{\rm ej} / \dot{M}_{\rm ej}$" + r" [erg M$_{\odot}^{-1}$]",
    yscale="log",
    ylim=(5e45, 2e49),
  
)
ax[1, 0].set(
    ylabel=r"$\dot{E}_{\rm  cooling} / \dot{M}_{\rm cooling}$" + r" [erg M$_{\odot}^{-1}$]",
    yscale="log",
    ylim=(5e45, 2e49),
  
  
)
ax[1, 1].set(
    ylabel=r"$\dot{E}_{\rm in} / \dot{M}_{\rm  in}$" + r" [erg M$_{\odot}^{-1}$]",
    yscale="log",
    ylim=(5e45, 2e49),
   
  
)
ax[1, 2].set(
    ylabel=r"$\dot{E}_{\rm ISM, wind} / \dot{M}_{\rm ISM, wind}$" + r" [erg M$_{\odot}^{-1}$]",
    yscale="log",
     ylim=(5e45, 2e49),
    
   
)   

# choose the redshift ticks you want to draw (edit this list)
z_tick = [20, 12, 8, 7,6, 5, 4]

# convert chosen z ticks to cosmic-time positions
t_ticks = np.array([LCDM.age(z).to_value(u.Gyr) for z in z_tick])
z_tick = np.array(z_tick)

# keep only ticks inside the plotted time range
mask = (t_ticks >= t_span[0]) & (t_ticks <= t_span[1])
t_ticks = t_ticks[mask]
z_tick = z_tick[mask]

# ensure x positions are in increasing order
order = np.argsort(t_ticks)
t_ticks = t_ticks[order]
z_tick = z_tick[order]

for col in range(3):
    ax2 = ax[0, col].twiny()
    ax2.set_xlim(ax[0, col].get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels([f"{z:.1f}" for z in z_tick])
    ax2.set_xlabel(r"$z$", labelpad=8)
    ax2.minorticks_off()
    ax[0, col].minorticks_on()
    
for axes in ax.flatten():
    for line in axes.lines:
        line.set_zorder(1)
        
        
ax[0, 1].legend(
    frameon=False,
    ncols=8,
    
    title=r"$f_{\mathrm{prevent,~floor}}$",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.15),
    title_fontsize=10,
)
# add anotation for z0 halo mass
ax[0, 0].text(
    0.95,
    0.95,
    z0_text,
    transform=ax[0, 0].transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=1),
    ha="right",
    va="top",
)

# add axis labels
ax_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
for i, axes in enumerate(ax.flatten()):
    axes.text(
        0.05,
        0.95,
        ax_labels[i],
        transform=axes.transAxes,
        fontsize=11,
        ha="left",
        va="top",
    )

# timescales plots
ax[2, 0].set(
    ylabel=r"$t_{\rm cool}$ [Gyr]",
    yscale="log",
    ylim=(5e-4, 0.9),
    xlabel=r"$t_{\rm univ}$ [Gyr]",
    xlim=t_span,
)
ax[2, 1].set(ylabel=r"$\dot{M}_{\rm in}$ [M$_{\odot}$ Gyr$^{-1}$]", yscale="log", xlabel=r"$t_{\rm univ}$ [Gyr]",)
ax[2, 2].set(ylabel=r"$t_{\rm dep, eff}$ [Gyr]", yscale="log", xlabel=r"$t_{\rm univ}$ [Gyr]",  ylim=(5e-4, 0.9))

plt.savefig(
    "./final_figs/appendix_energy_fprevent_Mhalo{:.2e}.pdf".format(mhalo_0.value), dpi=200, bbox_inches="tight", pad_inches=0.05
)
plt.show()
# %%
