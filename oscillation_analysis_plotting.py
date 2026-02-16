# %%

import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import cosmology
import astropy.units as u
import cgm_sf_regulator
from cgm_sf_regulator import CGMRegulator
from cgm_sf_regulator_baseline import CGMRegulatorBaseline
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit

importlib.reload(cgm_sf_regulator)
import pltstyle

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
)  # to apply the custom matplotlib style

red1 = "tab:red"
red2 = "tab:orange"
blu1 = "dodgerblue"
blu2 = "tab:green"


# standard flat cosmology
H0 = 70
Omegam0 = 0.3
Omegade0 = 0.7

Ob0 = 0.0490
f_b = Ob0 / Omegam0
LCDM = cosmology.LambdaCDM(H0=H0, Om0=Omegam0, Ode0=Omegade0)

mhalo_z0 = 1e12 * u.Msun
t_span = (0.1, 13.3)  # gyrs

# %% do a single run of the 2 phase model and breakdown the energy and mass contributions
## this is figure 2 of the first draft of the SF paper

model_2phase = CGMRegulator(
    mhalo_z0,
    t_span,
    tstep=0.005,
    add_f_prevent_floor=1e-6,  # virtually no floor
    KS_kappa_s=0.0464,
    KS_n=1.8,
    disk_scale_length=0.02,
    KS_parametrization="KS1998",
    TEST_tej_Tvir_definition=True,
)
run_2phase = model_2phase.run_halo()
results_2phase = model_2phase.get_results()
derived_2phase = model_2phase.get_derived_quantities()

tmin = results_2phase["t"].min() * 0.1
tmax = results_2phase["t"].max()

print(derived_2phase.keys())
print(results_2phase.keys())
# %%mv
fig, ax = plt.subplots(
    4,
    2,
    figsize=(10, 9),
    dpi=300,
    gridspec_kw={"height_ratios": [2.5, 0.5, 3, 3]},
)
plt.subplots_adjust(hspace=0.05, wspace=0.2)
ax = ax.flatten()

cmap = plt.get_cmap("Dark2")
hot_clr = cmap(1)
col_clr = cmap(0)

cmapdark2 = plt.get_cmap("Dark2")
total_clr = cmapdark2(0)

# Panel 0: CGM mass rates
# let's also include the negative values and take absolute values and have that as dotted line
# neg_dot_m_cgm_hot_mask = derived_2phase["dot_m_cgm_hot"] < 0
# neg_dot_m_cgm_cold_mask = derived_2phase["dot_m_cgm_cold"] < 0

time_hot = results_2phase["t"]  # [neg_dot_m_cgm_hot_mask]
time_cold = results_2phase["t"]  # [neg_dot_m_cgm_cold_mask]

y_hot = -derived_2phase["dot_m_cgm_hot"]  # [neg_dot_m_cgm_hot_mask]
y_cold = -derived_2phase["dot_m_cgm_cold"]  # [neg_dot_m_cgm_cold_mask]

ax[0].plot(time_hot, y_hot, color=hot_clr, lw=2, ls=":", label=" ")

ax[0].plot(time_cold, y_cold, color=col_clr, lw=2, ls=":", label=" ")

ax[4].plot(time_hot, y_hot, color=hot_clr, lw=2, ls=":", label="")

ax[4].plot(time_cold, y_cold, color=col_clr, lw=2, ls=":", label="")

ax[0].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_hot"],
    color=hot_clr,
    lw=2,
    label=r"hot",
    alpha=0.8,
)
ax[0].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_cold"],
    color=col_clr,
    lw=2,
    label=r"cold",
    alpha=0.8,
)

ax[0].legend(frameon=False, ncol=2, loc="lower right", fontsize=10)
ax[0].set(
    ylabel=r"$\dot{M}_{\rm CGM}  ~ [{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
    yscale="log",
    ylim=(2e5, 3e10),
)
ax[0].text(
    0.63,
    0.33,
    r"($-$)",
    transform=ax[0].transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="k",
)
ax[0].text(
    0.85,
    0.33,
    r"(+)",
    transform=ax[0].transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="k",
)

# Panel 3: CGM mass rates (inset range)
ax[4].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_hot"],
    color=hot_clr,
    lw=2,
    alpha=0.8,
    label=r"$\dot{M}_{\rm CGM, hot}$",
)
ax[4].plot(
    results_2phase["t"],
    derived_2phase["dot_m_cgm_cold"],
    color=col_clr,
    lw=2,
    alpha=0.8,
    label=r"$\dot{M}_{\rm CGM, cold}$",
)
ax[4].set(yscale="log", xlim=(0.05, 1.4), ylim=(4e5, None))

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
ax[1].set(
    ylabel=r"$\dot{E}_{\rm CGM}~ [{\rm erg\: Gyr^{-1}}]$",
    yscale="log",
    ylim=(2e52, None),
)
# ax[1].legend(frameon=False, ncol=1, loc="lower right")

# Panel 4: CGM energy rates (inset range)
ax[5].plot(
    results_2phase["t"],
    dot_e_cgm_total,
    color=hot_clr,
    lw=2,
    label=r"$\dot{E}_{\rm CGM}$",
)
ax[5].set(yscale="log", xlim=(0.05, 1.4), ylim=(5e52, None))

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
    xyB=(1.4, 5e52),
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
    xyB=(1.4, 4e5),
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

ax[6].plot(
    results_2phase["t"],
    dot_m_cgm_ej,
    lw=2,
    ls="--",
    label=r"$\dot{M}_{\rm CGM, ej}$",
    color=blu2,
)
ax[6].plot(
    results_2phase["t"],
    dot_cgm_falling,
    lw=2,
    ls="--",
    label=r"$\dot{M}_{\rm CGM, falling}$",
    color="y",
)

ax[6].plot(
    results_2phase["t"],
    dot_m_cgm_in,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, in}$",
    color=red1,
)
ax[6].plot(
    results_2phase["t"],
    dot_m_sne_wind,
    lw=2,
    label=r"$\dot{M}_{\rm SNe, wind}$",
    color=red2,
)


ax[6].set(
    ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
    yscale="log",
    ylim=(3e5, 3e10),
    xlabel=r"$t_{\rm univ}$ [Gyr]",
    xlim=(0.05, 1.4),
)
ax[6].plot(
    results_2phase["t"],
    dot_m_cgm_cooling,
    lw=2,
    ls="-",
    label=r"$\dot{M}_{\rm CGM, cooling}$",
    color=blu1,
)
ax[6].legend(
    frameon=False, ncol=3, fontsize=10, loc="lower right", bbox_to_anchor=(1.1, -0.6)
)

ax[6].text(
    0.15,
    -0.25,
    r"removes gas",
    transform=ax[6].transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="grey",
)
ax[6].text(
    0.5,
    -0.25,
    r"adds gas",
    transform=ax[6].transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="grey",
)
ax[6].text(
    0.9,
    -0.25,
    "converts hot to cold",
    transform=ax[6].transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="grey",
)

# Panel 5: CGM energy rates (zoomed)
ax[7].plot(
    results_2phase["t"],
    dot_e_cgm_acc,
    color=red1,
    lw=2,
    label=r"$\dot{E}_{\rm CGM, in}$",
)
ax[7].plot(
    results_2phase["t"],
    dot_e_sne_wind,
    color=red2,
    lw=2,
    label=r"$\dot{E}_{\rm SNe, wind}$",
)
ax[7].plot(
    results_2phase["t"],
    dot_e_cgm_ej,
    color=blu2,
    lw=2,
    ls="--",
    label=r"$\dot{E}_{\rm CGM, ej}$",
)
ax[7].plot(
    results_2phase["t"],
    dot_e_cgm_cool,
    color=blu1,
    ls="--",
    lw=2,
    label=r"$\dot{E}_{\rm CGM, cooling}$",
)
ax[7].set(
    ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
    yscale="log",
    ylim=(5e52, 1e58),
    xlim=(0.05, 1.4),
    xlabel=r"$t_{\rm univ}$ [Gyr]",
)
ax[7].legend(
    frameon=False, ncol=2, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.6)
)

# first column of the legend add energy to the CGM , 2nd removes, add text annotation
ax[7].text(
    0.3,
    -0.25,
    r"adds energy",
    transform=ax[7].transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="grey",
)
ax[7].text(
    0.6,
    -0.25,
    r"removes energy",
    transform=ax[7].transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="grey",
)

# make a twin redshift axis for the top row, using z
t_ticks = np.array([1, 3, 5, 8, 10, 12, 13.3])
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

t_ticks = np.array([0.3, 0.6, 1, 1.4])
z_ticks = [cosmology.z_at_value(LCDM.age, t * u.Gyr).value for t in t_ticks]
for i in [4, 5]:
    ax2 = ax[i].twiny()
    ax2.plot(results_2phase["t"], derived_2phase["dot_m_cgm_hot"], color="k", alpha=0)
    ax2.set_xlim((0.05, 1.4))
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels(["{:.1f}".format(z) for z in z_ticks])
    # ax2.set_xlabel(r"$z$")
    ax2.minorticks_off()
    ax[i].minorticks_on()
    ax[i].set_xlim((0.05, 1.4))
    # turn x axis ticks off
    ax[i].set_xticks([])

for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)

# plt.savefig(
#     "./figures/fig_2phase_detailed_cgm_1e12.png",
#     dpi=200,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
plt.show()


# %% zoom in on the oscillations, same as above but with annotation
### Figure 3 of latest draft

model_2phase = CGMRegulator(
    mhalo_z0,
    (0.15, 1),
    tstep=0.001,
    add_f_prevent_floor=1e-6,  # virtually no floor
    KS_kappa_s=0.0464,
    KS_n=1.8,
    disk_scale_length=0.02,
    KS_parametrization="KS1998",
    TEST_tej_Tvir_definition=True,
)
run_2phase = model_2phase.run_halo()
results_2phase = model_2phase.get_results()
derived_2phase = model_2phase.get_derived_quantities()

model_2Phase_long = CGMRegulator(
    mhalo_z0,
    (0.15, 13),
    add_f_prevent_floor=1e-6,  # virtually no floor
    KS_kappa_s=0.0464,
    KS_n=1.8,
    disk_scale_length=0.02,
    KS_parametrization="KS1998",
    TEST_tej_Tvir_definition=True,
)
run_2Phase_long = model_2Phase_long.run_halo()
results_2Phase_long = model_2Phase_long.get_results()
derived_2Phase_long = model_2Phase_long.get_derived_quantities()

# %%

xlim_for_zoom = (0.166, 0.35)
fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.23, wspace=0.2)

# add some inset panels for long term evolution
ax0 = ax[0].inset_axes([0, 1.2, 0.45, 1.4])
ax1 = ax[1].inset_axes([0.55, 2.44, 0.45, 1.4])
dot_m_cgm_cooling = results_2Phase_long["m_cgm_hot"] / derived_2Phase_long["tcool_real"]
dot_m_cgm_in = derived_2Phase_long["dot_m_cgm_in"]
dot_m_sne_wind = derived_2Phase_long["dot_m_ism_wind"]
dot_m_cgm_ej = derived_2Phase_long["dot_m_cgm_out"]
dot_cgm_falling = results_2Phase_long["m_cgm_cold"] / derived_2Phase_long["t_dynamical"]
dot_m_sfr = derived_2Phase_long["dot_m_sfr"]
ax0.plot(
    results_2Phase_long["z"],
    dot_m_sfr,
    lw=2,
    label=r"$\dot{M}_{\rm \star}$",
    color="mediumorchid",
)
ax0.plot(
    results_2Phase_long["z"],
    dot_m_cgm_ej,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, ej}$",
    color=blu2,
)
ax0.plot(
    results_2Phase_long["z"],
    dot_cgm_falling,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, falling}$",
    color="y",
)
ax0.plot(
    results_2Phase_long["z"],
    dot_m_cgm_cooling,
    lw=2,
    label=r"$\dot{M}_{\rm CGM, cooling}$",
    color=blu1,
)
ax0.plot(
    results_2Phase_long["z"],
    dot_m_cgm_in,
    ls="--",
    lw=2,
    label=r"$\dot{M}_{\rm CGM, in}$",
    color=red1,
)
ax0.plot(
    results_2Phase_long["z"],
    dot_m_sne_wind,
    ls="--",
    lw=2,
    label=r"$\dot{M}_{\rm SNe, wind}$",
    color=red2,
)
ax0.set(
    ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
    yscale="log",
    ylim=(1e7, 2e10),
    # xscale="log",
    xlabel=r"$z$",
    xlim=(0, cosmology.z_at_value(LCDM.age, xlim_for_zoom[0] * u.Gyr).value),
)
# flip x axis
ax0.invert_xaxis()

# set the x axis to top
ax0.xaxis.set_label_position("top")
ax0.xaxis.tick_top()
ax0.xaxis.set_tick_params(labeltop=True)
ax0.xaxis.labelpad = 10

dot_e_sne_wind = derived_2Phase_long["dot_e_ism_wind"]
dot_e_cgm_acc = derived_2Phase_long["dot_e_cgm_in"]
dot_e_cgm_cool = derived_2Phase_long["dot_e_cgm_cooling"]
dot_e_cgm_ej = derived_2Phase_long["dot_e_cgm_out"]

ax1.plot(
    results_2Phase_long["z"],
    dot_e_cgm_ej,
    color=blu2,
    lw=2,
    label=r"$\dot{E}_{\rm CGM, ej}$",
)
ax1.plot(
    results_2Phase_long["z"],
    dot_e_cgm_cool,
    color=blu1,
    lw=2,
    label=r"$\dot{E}_{\rm CGM, cooling}$",
)
ax1.plot(
    results_2Phase_long["z"],
    dot_e_cgm_acc,
    color=red1,
    ls="--",
    lw=2,
    label=r"$\dot{E}_{\rm CGM, in}$",
)
ax1.plot(
    results_2Phase_long["z"],
    dot_e_sne_wind,
    color=red2,
    ls="--",
    lw=2,
    label=r"$\dot{E}_{\rm SNe, wind}$",
)
ax1.set(
    ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
    yscale="log",
    ylim=(9e53, 9e57),
    # xscale="log",
    xlabel=r"$z$",
    xlim=(0.0, cosmology.z_at_value(LCDM.age, xlim_for_zoom[0] * u.Gyr).value),
)
# set the x axis to top
ax1.xaxis.set_label_position("top")
ax1.xaxis.tick_top()
ax1.xaxis.set_tick_params(labeltop=True)
ax1.invert_xaxis()
# pad x label
ax1.xaxis.labelpad = 10

dot_m_cgm_cooling = results_2phase["m_cgm_hot"] / derived_2phase["tcool_real"]
dot_m_cgm_in = derived_2phase["dot_m_cgm_in"]
dot_m_sne_wind = derived_2phase["dot_m_ism_wind"]
dot_m_cgm_ej = derived_2phase["dot_m_cgm_out"]
dot_cgm_falling = results_2phase["m_cgm_cold"] / derived_2phase["t_dynamical"]
dot_m_sfr = derived_2phase["dot_m_sfr"]
f_prevent = derived_2phase["dot_e_cgm_cooling"] / derived_2phase["dot_e_cgm_out"]

red1 = "tab:red"
red2 = "tab:orange"
blu1 = "dodgerblue"
blu2 = "tab:green"

ax[0].plot(
    results_2phase["t"],
    dot_m_sfr,
    lw=3,
    label=r"$\dot{M}_{\rm \star}$",
    color="mediumorchid",
)
ax[0].plot(
    results_2phase["t"],
    dot_m_cgm_ej,
    lw=3,
    label=r"$\dot{M}_{\rm CGM, ej}$",
    color=blu2,
)
ax[0].plot(
    results_2phase["t"],
    dot_cgm_falling,
    lw=3,
    label=r"$\dot{M}_{\rm CGM, falling}$",
    color="y",
)
ax[0].plot(
    results_2phase["t"],
    dot_m_cgm_cooling,
    lw=3,
    label=r"$\dot{M}_{\rm CGM, cooling}$",
    color=blu1,
)
ax[0].plot(
    results_2phase["t"],
    dot_m_cgm_in,
    ls="--",
    lw=3,
    label=r"$\dot{M}_{\rm CGM, in}$",
    color=red1,
)
ax[0].plot(
    results_2phase["t"],
    dot_m_sne_wind,
    ls="--",
    lw=3,
    label=r"$\dot{M}_{\rm SNe, wind}$",
    color=red2,
)
ax[0].set(
    ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
    yscale="log",
    ylim=(4e5, 2e9),
    # xlabel=r"$t_{\rm univ}$ [Gyr]",
)
ax[0].legend(
    frameon=False,
    loc="upper center",
    fontsize=10.5,
    ncol=6,
    bbox_to_anchor=(0.5, 1.22),
)

# add a grey shaded region in the long runs
ax0.axvspan(
    cosmology.z_at_value(LCDM.age, xlim_for_zoom[0] * u.Gyr).value,
    cosmology.z_at_value(LCDM.age, xlim_for_zoom[1] * u.Gyr).value,
    facecolor="grey",
    alpha=0.2,
)

# Panel 1: CGM energy rates
dot_e_sne_wind = derived_2phase["dot_e_ism_wind"]
dot_e_cgm_acc = derived_2phase["dot_e_cgm_in"]
dot_e_cgm_cool = derived_2phase["dot_e_cgm_cooling"]
dot_e_cgm_ej = derived_2phase["dot_e_cgm_out"]
dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool
ax[1].plot(
    results_2phase["t"],
    dot_e_cgm_ej,
    color=blu2,
    lw=3,
    label=r"$\dot{E}_{\rm CGM, ej}$",
)
ax[1].plot(
    results_2phase["t"],
    dot_e_cgm_cool,
    color=blu1,
    lw=3,
    label=r"$\dot{E}_{\rm CGM, cooling}$",
)
ax[1].plot(
    results_2phase["t"],
    dot_e_cgm_acc,
    color=red1,
    ls="--",
    lw=3,
    label=r"$\dot{E}_{\rm CGM, in}$",
)
ax[1].plot(
    results_2phase["t"],
    dot_e_sne_wind,
    ls="--",
    color=red2,
    lw=3,
    label=r"$\dot{E}_{\rm SNe, wind}$",
)

ax[1].set(
    ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
    yscale="log",
    ylim=(5e52, 1e57),
    xlim=xlim_for_zoom,
    xlabel=r"$t_{\rm univ}$ [Gyr]",
)
ax[1].legend(
    frameon=False,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.2),
    fontsize=10.5,
)

ax1.axvspan(
    cosmology.z_at_value(LCDM.age, xlim_for_zoom[0] * u.Gyr).value,
    cosmology.z_at_value(LCDM.age, xlim_for_zoom[1] * u.Gyr).value,
    facecolor="grey",
    alpha=0.2,
)

for axes in ax:
    for line in axes.lines:
        line.set_zorder(1)


#### add key cycle points
x = [0.19, 0.223, 0.25, 0.27, 0.305]
label = ["(a)", "(b)", "(c)", "(d)", "(e)"]
colors = ["k", "k", "k", "k", "k"]
ty = [1e9, 1e9, 1e6, 5e6, 5e6]  # text y positions

for xi, lab, col, text_y in zip(x, label, colors, ty):
    # draw vertical line in two segments, leaving a gap for the label

    gap = 0.35 * text_y  # size of the gap around the label
    y1, y2 = 4e5, text_y - gap  # lower segment up to just below the label
    y3, y4 = text_y + gap, 2e9  # upper segment from just above the label to top

    ax[0].plot([xi, xi], [y1, y2], color=col, lw=3, alpha=0.7, zorder=10)
    ax[0].plot([xi, xi], [y3, y4], color=col, lw=3, alpha=0.7, zorder=10)

    ax[1].axvline(x=xi, color=col, lw=3, alpha=0.7, zorder=10)

    # place label in the gap

    ax[0].text(xi, text_y, lab, ha="center", va="center", fontsize=10, zorder=11)

# make the background color grey
ax[0].set_facecolor("whitesmoke")
ax[1].set_facecolor("whitesmoke")

# plt.savefig(
#     "./figures/fig_cgm_cycle_zoom.png", dpi=200, bbox_inches="tight", pad_inches=0.05
# )
plt.show()

# %% ######### now phase plot of the cycle this did not do anything


# we start by fitting the oscillations to get some central tendency
def ln_fit(x, a, b):
    return a * np.log(x) + b


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
        "xtick.major.size": 3,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

dot_total_x = []
dot_total_y = []
dot_total_x_ln = []
dot_e_add_x = []
dot_e_add_y = []
dot_e_add_x_ln = []
dot_e_sub_x = []
dot_e_sub_y = []
dot_e_sub_x_ln = []
halo_masses = np.array([1e11, 1e12, 1e13]) * u.Msun

for i, Mh_z0 in enumerate(halo_masses):
    model_2phase = CGMRegulator(
        Mh_z0,
        (0.15, 0.8),
        tstep=0.0005,
        add_f_prevent_floor=1e-6,  # virtually no floor
        KS_kappa_s=0.0464,
        KS_n=1.8,
        disk_scale_length=0.02,
        KS_parametrization="KS1998",
        TEST_tej_Tvir_definition=True,
    )
    run_2phase = model_2phase.run_halo()
    results_2phase = model_2phase.get_results()
    derived_2phase = model_2phase.get_derived_quantities()

    mask = (0.25, 1)
    t_mask = (results_2phase["t"] > mask[0]) & (results_2phase["t"] < mask[1])
    dot_e_sne_wind = derived_2phase["dot_e_ism_wind"]
    dot_e_cgm_acc = derived_2phase["dot_e_cgm_in"]
    dot_e_cgm_cool = derived_2phase["dot_e_cgm_cooling"]
    dot_e_cgm_ej = derived_2phase["dot_e_cgm_out"]
    dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool
    dot_e_add = dot_e_cgm_acc + dot_e_sne_wind
    dot_e_sub = dot_e_cgm_ej + dot_e_cgm_cool
    dot_e_cgm_total_masked = dot_e_cgm_total[t_mask]

    E_cgm_masked = results_2phase["egy_cgm"][t_mask]
    E_ism_masked = results_2phase["egy_ism_wind"][t_mask]
    E_radloss_masked = results_2phase["egy_radloss"][t_mask]
    E_eject_masked = results_2phase["egy_eject"][t_mask]
    E_accrete_masked = results_2phase["egy_accrete"][t_mask]

    t = results_2phase["t"][t_mask]
    z = results_2phase["z"][t_mask]
    dot_e_add = dot_e_add[t_mask]
    dot_e_sub = dot_e_sub[t_mask]

    # bin the E_cgm_total and find the average in each bin using scipy stats
    mean_bin_size = 0.01  # Gyr
    bin_edges = np.arange(t.min(), t.max(), mean_bin_size)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    E_cgm_binned = binned_statistic(
        t, E_cgm_masked, statistic="mean", bins=bin_edges
    ).statistic

    # Fit a polynomial of degree 3 (change degree as needed)
    poly_deg = 6
    coeffs = np.polyfit(bin_centers, E_cgm_binned, poly_deg)
    E_cgm_polyfit = np.polyval(
        coeffs, t
    )  # find corresponding E_cgm values using the fit

    # Only use bins where both x and y are positive
    valid = (bin_centers > 0) & (E_cgm_binned > 0)
    x_fit = bin_centers[valid]
    y_fit = E_cgm_binned[valid]

    # Fit in log-log space
    popt, pcov = curve_fit(ln_fit, x_fit, np.log(y_fit))
    a_fit, b_fit = popt

    # For plotting: fitted curve in original space
    x_fit_full = np.linspace(mask[0], mask[1], 100)
    E_cgm_fit_full = np.exp(ln_fit(x_fit_full, a_fit, b_fit))

    E_cgm_fit_ln = np.exp(ln_fit(t, a_fit, b_fit))

    # first plot E_cgm as a function of time to see the oscillations
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
    ax.scatter(
        bin_centers, E_cgm_binned, color="tab:blue", label="binned mean", alpha=0.7
    )
    ax.plot(
        x_fit_full, E_cgm_fit_full, color="tab:blue", lw=3, label="ln fit to binned"
    )

    ax.plot(t, E_ism_masked, color="mediumorchid", lw=2, label=r"$E_{\rm SNe, wind}$")
    ax.plot(t, E_radloss_masked, color="tab:blue", lw=2, label=r"$E_{\rm radloss}$")
    ax.plot(t, E_eject_masked, color="tab:green", lw=2, label=r"$E_{\rm eject}$")
    ax.plot(t, E_accrete_masked, color="tab:red", lw=2, label=r"$E_{\rm accrete}$")

    egy_additive = E_ism_masked + E_accrete_masked
    egy_subtractive = E_eject_masked + E_radloss_masked

    ax.plot(
        t,
        egy_additive,
        lw=2,
        label=r"$E_{\rm eject} + E_{\rm radloss}$",
        ls="--",
    )
    ax.plot(
        t,
        egy_subtractive,
        lw=2,
        label=r"$E_{\rm SNe, wind} + E_{\rm accrete}$",
        ls="--",
    )
    # make the same polyfit for egy_additive and egy_subtractive
    # Polyfit for egy_additive
    coeffs_add = np.polyfit(
        bin_centers,
        binned_statistic(t, egy_additive, statistic="mean", bins=bin_edges).statistic,
        poly_deg,
    )
    egy_additive_polyfit = np.polyval(coeffs_add, t)

    # Polyfit for egy_subtractive
    coeffs_sub = np.polyfit(
        bin_centers,
        binned_statistic(
            t, egy_subtractive, statistic="mean", bins=bin_edges
        ).statistic,
        poly_deg,
    )
    egy_subtractive_polyfit = np.polyval(coeffs_sub, t)

    # ln fit for additive
    valid_add = (bin_centers > 0) & (
        binned_statistic(t, egy_additive, statistic="mean", bins=bin_edges).statistic
        > 0
    )
    x_fit_add = bin_centers[valid_add]
    y_fit_add = binned_statistic(
        t, egy_additive, statistic="mean", bins=bin_edges
    ).statistic[valid_add]
    popt_add, _ = curve_fit(ln_fit, x_fit_add, np.log(y_fit_add))
    a_fit_add, b_fit_add = popt_add
    egy_additive_fit_ln = np.exp(ln_fit(t, a_fit_add, b_fit_add))
    ax.plot(
        t,
        egy_additive_fit_ln,
        color="tab:red",
        lw=2,
        alpha=0.7,
        ls=":",
        label="ln fit add",
    )

    # ln fit for subtractive
    valid_sub = (bin_centers > 0) & (
        binned_statistic(t, egy_subtractive, statistic="mean", bins=bin_edges).statistic
        > 0
    )
    x_fit_sub = bin_centers[valid_sub]
    y_fit_sub = binned_statistic(
        t, egy_subtractive, statistic="mean", bins=bin_edges
    ).statistic[valid_sub]
    popt_sub, _ = curve_fit(ln_fit, x_fit_sub, np.log(y_fit_sub))
    a_fit_sub, b_fit_sub = popt_sub
    egy_subtractive_fit_ln = np.exp(ln_fit(t, a_fit_sub, b_fit_sub))
    ax.plot(
        t,
        egy_subtractive_fit_ln,
        color="tab:blue",
        lw=2,
        alpha=0.7,
        ls=":",
        label="ln fit sub",
    )

    ax.plot(
        t, egy_additive_polyfit, color="tab:red", lw=2, alpha=0.3, label="polyfit add"
    )
    ax.plot(
        t,
        egy_subtractive_polyfit,
        color="tab:blue",
        lw=2,
        alpha=0.3,
        label="polyfit sub",
    )
    ax.plot(
        t,
        E_ism_masked - E_eject_masked - E_radloss_masked + E_accrete_masked,
        lw=2,
        label=r"$E_{\rm total}$",
        color="k",
    )
    ax.plot(t, E_cgm_polyfit, color="k", lw=5, alpha=0.3, label="polyfit full res")

    ax.set(
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        ylabel=r"$E_{\rm CGM}$ [erg]",
        yscale="log",
        xlim=mask,
        xscale="log",
    )
    ax.legend(ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01), loc="lower center")
    plt.show()
    x = E_cgm_masked - E_cgm_polyfit
    x_ln = E_cgm_masked - E_cgm_fit_ln
    y = dot_e_cgm_total_masked
    x2 = egy_additive - egy_additive_polyfit
    x2_ln = egy_additive - egy_additive_fit_ln
    y2 = dot_e_add
    x3 = egy_subtractive - egy_subtractive_polyfit
    x3_ln = egy_subtractive - egy_subtractive_fit_ln
    y3 = dot_e_sub
    dot_total_x.append(x)
    dot_total_x_ln.append(x_ln)
    dot_total_y.append(y)
    dot_e_add_x.append(x2)
    dot_e_add_x_ln.append(x2_ln)
    dot_e_add_y.append(y2)
    dot_e_sub_x.append(x3)
    dot_e_sub_x_ln.append(x3_ln)
    dot_e_sub_y.append(y3)


# %% filed phase plot
fig, ax = plt.subplots(3, 3, figsize=(10, 10), dpi=300)
plt.subplots_adjust(hspace=0.1, wspace=0.1)
mass_labels = [
    r"$ 10^{11}\,{\rm M_{\odot}}$",
    r"$ 10^{12}\,{\rm M_{\odot}}$",
    r"$ 10^{13}\,{\rm M_{\odot}}$",
]

for row, Mh_z0 in enumerate(halo_masses):
    # add text label for each row
    ax[row, 0].text(
        0.05,
        0.25,
        mass_labels[row],
        transform=ax[row, 0].transAxes,
        fontsize=12,
        ha="left",
        va="bottom",
        color="k",
    )

    # x = dot_total_x[row]
    y = dot_total_y[row]
    # x2 = dot_e_sub_x[row]
    y2 = dot_e_sub_y[row]
    # x3 = dot_e_add_x[row]
    y3 = dot_e_add_y[row]

    x = dot_total_x_ln[row]
    x2 = dot_e_sub_x_ln[row]
    x3 = dot_e_add_x_ln[row]

    # --- First column: E_CGM phase plot ---
    ax[row, 0].plot(x, y, lw=1.5, color="k", alpha=0.5)
    ax[row, 0].plot(
        x[0],
        y[0],
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="k",
    )
    ax[row, 0].plot(
        x[-1], y[-1], marker="o", markersize=5, markerfacecolor="k", markeredgecolor="k"
    )
    N = 100
    for i in range(N, len(x), N):
        ax[row, 0].annotate(
            "",
            xy=(x[i], y[i]),
            xytext=(x[i - 1], y[i - 1]),
            arrowprops=dict(arrowstyle="->", color="k", lw=1),
        )

    ax[row, 0].spines["left"].set_position("center")
    ax[row, 0].spines["bottom"].set_position("center")
    ax[row, 0].spines["right"].set_color("none")
    ax[row, 0].spines["top"].set_color("none")
    ax[row, 0].xaxis.set_ticks_position("bottom")
    ax[row, 0].yaxis.set_ticks_position("left")
    ax[row, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    offset_text = ax[row, 0].yaxis.get_offset_text()
    offset_text.set_position((0.5, 1.02))

    # --- Second column: egy_subtractive phase plot ---
    ax[row, 1].plot(x2, y2, lw=1.5, color="tab:blue", alpha=0.7)
    ax[row, 1].plot(
        x2[0],
        y2[0],
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="tab:blue",
    )
    ax[row, 1].plot(
        x2[-1],
        y2[-1],
        marker="o",
        markersize=5,
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
    )
    for i in range(N, len(x2), N):
        ax[row, 1].annotate(
            "",
            xy=(x2[i], y2[i]),
            xytext=(x2[i - 1], y2[i - 1]),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1),
        )

    ax[row, 1].spines["left"].set_position("center")
    ax[row, 1].spines["bottom"].set_position("center")
    ax[row, 1].spines["right"].set_color("none")
    ax[row, 1].spines["top"].set_color("none")
    ax[row, 1].xaxis.set_ticks_position("bottom")
    ax[row, 1].yaxis.set_ticks_position("left")
    ax[row, 1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    offset_text = ax[row, 1].yaxis.get_offset_text()
    offset_text.set_position((0.5, 1.02))

    # --- Third column: egy_additive phase plot ---
    ax[row, 2].plot(x3, y3, lw=1.5, color="tab:red", alpha=0.7)
    ax[row, 2].plot(
        x3[0],
        y3[0],
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="tab:red",
    )
    ax[row, 2].plot(
        x3[-1],
        y3[-1],
        marker="o",
        markersize=5,
        markerfacecolor="tab:red",
        markeredgecolor="tab:red",
    )
    for i in range(N, len(x3), N):
        ax[row, 2].annotate(
            "",
            xy=(x3[i], y3[i]),
            xytext=(x3[i - 1], y3[i - 1]),
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1),
        )

    ax[row, 2].spines["left"].set_position("center")
    ax[row, 2].spines["bottom"].set_position("center")
    ax[row, 2].spines["right"].set_color("none")
    ax[row, 2].spines["top"].set_color("none")
    ax[row, 2].xaxis.set_ticks_position("bottom")
    ax[row, 2].yaxis.set_ticks_position("left")
    ax[row, 2].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    offset_text = ax[row, 2].yaxis.get_offset_text()
    offset_text.set_position((0.5, 1.02))

    ax[row, 0].set_xlim(np.abs(x).max() * np.array([-1, 1]) * 1.05)
    ax[row, 1].set_xlim(np.abs(x2).max() * np.array([-1, 1]) * 1.05)
    ax[row, 2].set_xlim(np.abs(x3).max() * np.array([-1, 1]) * 1.05)
    ax[row, 0].set_ylim(np.abs(y).max() * np.array([-1, 1]) * 1.05)
    ax[row, 1].set_ylim(np.abs(y2).max() * np.array([-1, 1]) * 1.05)
    ax[row, 2].set_ylim(np.abs(y3).max() * np.array([-1, 1]) * 1.05)


# write the run time
t_min, t_max = results_2phase["t"][t_mask].min(), results_2phase["t"][t_mask].max()
z_max, z_min = results_2phase["z"][t_mask].max(), results_2phase["z"][t_mask].min()
ax[0, 0].text(
    -0.05,
    1.05,
    r"$t={:.2f}- {:.2f}$ Gyr".format(t_min, t_max)
    + "\n"
    + r"$z={:.2f}- {:.2f}$".format(z_max.value, z_min.value),
    transform=ax[0, 0].transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="k"),
)


ax[1, 0].set_ylabel(r"$\dot{E}$ [erg Gyr$^{-1}$]", labelpad=80)
# ax[0, 0].set_ylabel(r"$\dot{E}$ [erg Gyr$^{-1}$]", labelpad=80)
# ax[2, 0].set_ylabel(r"$\dot{E}$ [erg Gyr$^{-1}$]", labelpad=80)
ax[2, 1].set_xlabel(r"$E - \langle E\rangle_{\rm fit}$ [erg]", labelpad=80)

ax[0, 0].text(
    0.5,
    1.1,
    r"$E_{\rm CGM, total}$",
    transform=ax[0, 0].transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
    # bbox=dict(facecolor="white", alpha=0.8, edgecolor="k"),
)
ax[0, 1].text(
    0.5,
    1.1,
    r"$E_{\rm CGM, ej} + E_{\rm CGM, radloss}$",
    transform=ax[0, 1].transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
    # bbox=dict(facecolor="white", alpha=0.8, edgecolor="k"),
)
ax[0, 2].text(
    0.5,
    1.1,
    r"$E_{\rm SNe\, wind} + E_{\rm CGM, in}$",
    transform=ax[0, 2].transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
    # bbox=dict(facecolor="white", alpha=0.8, edgecolor="k"),
)

# plt.savefig(
#     "./figures/E_phase_plots.png", dpi=300, bbox_inches="tight", pad_inches=0.05
# )
plt.show()

# %%
