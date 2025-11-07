# %% here we scan through all 4 updates to the model and see their impact on the oscillations
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
import astropy.constants as consts
import astropy.units as u

importlib.reload(cgm_sf_regulator)
import pltstyle
from matplotlib.lines import Line2D

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
xlim_for_zoom = (0.166, 2)

# %% do a single run of the 2 phase model and breakdown the energy and mass contributions
## this is figure 2 of the first draft of the SF paper

min_fprevent = np.geomspace(0.01, 1, 20)


for fp in min_fprevent[:1:]:
    model_2phase = CGMRegulator(
        mhalo_z0,
        (0.2, xlim_for_zoom[1]),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        updated_halo_infall=True,
        updated_2phase_CGM=True,
        updated_SF_law=False,
        updated_loadings=True,
        alpha_e=0.1,
        alpha_m=0.1,
    )
    model_2Phase_long = CGMRegulator(
        mhalo_z0,
        (0.15, 13),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        updated_halo_infall=True,
        updated_2phase_CGM=True,
        updated_SF_law=False,
        updated_loadings=True,
        alpha_e=0.1,
        alpha_m=0.1,
    )

    run_2phase = model_2phase.run_halo()
    results_2phase = model_2phase.get_results()
    derived_2phase = model_2phase.get_derived_quantities()

    run_2Phase_long = model_2Phase_long.run_halo()
    results_2Phase_long = model_2Phase_long.get_results()
    derived_2Phase_long = model_2Phase_long.get_derived_quantities()

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.23, wspace=0.2)

    # add some inset panels for long term evolution
    ax0 = ax[0].inset_axes([0, 1.2, 0.45, 1.4])
    ax1 = ax[1].inset_axes([0.55, 2.44, 0.45, 1.4])
    dot_m_cgm_cooling = (
        results_2Phase_long["m_cgm_hot"] / derived_2Phase_long["tcool_real"]
    )
    dot_m_cgm_in = derived_2Phase_long["dot_m_cgm_in"]
    dot_m_sne_wind = derived_2Phase_long["dot_m_ism_wind"]
    dot_m_cgm_ej = derived_2Phase_long["dot_m_cgm_out"]
    dot_cgm_falling = (
        results_2Phase_long["m_cgm_cold"] / derived_2Phase_long["t_dynamical"]
    )
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
        label=r"$\dot{M}_{\rm CGM, acc}$",
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
        ylim=(1e7, None),
        # xscale="log",
        xlabel=r"$z$",
        xlim=(0, 15),
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
        label=r"$\dot{E}_{\rm CGM, acc}$",
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
        ylim=(5e54, None),
        # xscale="log",
        xlabel=r"$z$",
        xlim=(0.0, 15),
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
        label=r"$\dot{M}_{\rm CGM, acc}$",
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
        ylim=(4e5, None),
        # xlabel=r"$t_{\rm univ}$ [Gyr]",
    )
    ax[0].legend(
        frameon=False,
        loc="upper center",
        fontsize=10.5,
        ncol=6,
        bbox_to_anchor=(0.5, 1.22),
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
        label=r"$\dot{E}_{\rm CGM, acc}$",
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
        ylim=(5e52, None),
        xlim=xlim_for_zoom,
    )
    ax[1].legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fontsize=10.5,
    )
    for axes in ax:
        for line in axes.lines:
            line.set_zorder(1)

    # ===== add an inset axis to add a third row with the f_prevent evolution
    axin = ax[1].inset_axes([0, -1.2, 1, 1])
    axin.plot(
        results_2phase["t"],
        f_prevent,
        color="k",
        lw=3,
    )
    axin.set(
        ylabel=r"$f_{\rm prevent}$ unclipped",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        ylim=(5e-3, None),
        yscale="log",
    )
    axin.axhspan(fp, 1, color="k", alpha=0.1)

    # add another inset for the cgm temperature
    ax2 = ax[1].inset_axes([0, -2.22, 1, 1])
    ax2.plot(
        results_2phase["t"],
        derived_2phase["cgm_temp"],
        color="crimson",
        lw=3,
        label=r"$T_{\rm CGM}$",
    )

    # we can also plot the virial temperature for reference: halo_vir_temp
    ax2.plot(
        results_2phase["t"],
        derived_2phase["halo_vir_temp"],
        color="forestgreen",
        lw=3,
        ls="--",
        label=r"$T_{\rm vir}$",
    )
    ax2.legend(loc="lower right")

    ax2.set(
        ylabel=r"$\rm Temperature [K]$",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
    )
    axin.text(
        0.05,
        0.85,
        r"$f_{{\rm prevent, min}} = {:.6f}$".format(round(fp, 6)),
        transform=axin.transAxes,
        fontsize=12,
        ha="left",
        va="top",
        color="k",
    )

    # add the lambda cooling rate in another inset
    ax3 = ax[1].inset_axes([0, -3.24, 1, 1])
    ax3.plot(
        results_2phase["t"],
        derived_2phase["cooling_lambda"],
        color="darkcyan",
        lw=3,
    )
    ax3.set(
        ylabel=r"$\Lambda$ [erg cm$^3$ s$^{-1}$]",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
        ylim=(1e-25, 1e-20),
    )

    # add the mass of the halo at z = 0on the first panel
    ax[0].text(
        0.05,
        0.9,
        r"$M_{{\rm halo}}(z=0) = {:.1e} \: M_{{\odot}}$".format(mhalo_z0.value),
        transform=ax[0].transAxes,
        fontsize=12,
        ha="left",
        va="top",
        color="k",
    )
    # plt.savefig(
    # "./figures/f_prevent_zoom_test/2phase_detailed_cgm_1e12_fp_{:}.png".format(
    #     round(fp, 6)
    # ),
    # dpi=200,
    # bbox_inches="tight",
    # pad_inches=0.05,
    # )

    plt.show()
# %% what is so different with the infalls and loading reparametrizations that makes oscillations appear?


def custom_mass_loading(mhalo, A=10, alpha=-1.4):
    """mass loading factor as a function of halo mass"""
    return A * (mhalo / (1e10 * u.solMass)) ** alpha


def custom_energy_loading(mhalo_z0, A=0.10, alpha=-0.5):
    """energy loading fact or as a function of halo mass"""
    eta_e = A * (mhalo_z0 / (1e12 * u.solMass)) ** alpha
    if np.any(eta_e > 1):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


def vcirc_energy_loading(halo_vcirc, alpha_e=0.1):
    eta_e = alpha_e * (halo_vcirc.value / 200) ** (-3 / 2)

    # if eta e > 1 set to 1, halo_vcirc can be float or array
    if np.any(eta_e > 0):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


def vcirc_mass_loading(halo_vcirc, alpha_m=9):
    return alpha_m * (halo_vcirc.value / 200) ** (-3 / 2)


def virial_radius(z, mhalo, Delc=200):

    return (mhalo / (LCDM.critical_density(z) * (4 / 3) * np.pi * Delc)) ** (1 / 3)


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def virial_T(mhalo, Rvir):

    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return (2 / 5) * ((G * mhalo * mp) / (Rvir * kb))



# %%


def halo_infall_dekel(z, mhalo):

    mdot = (
        0.47
        * mhalo
        * (mhalo / (1e12 * u.solMass)) ** (0.15)
        * ((1 + z) / 3) ** (2.25)
        * u.Gyr ** (-1)
    ).to(u.solMass / u.Gyr)
    try:
        if mdot <= 0:
            print("____", mhalo)
    except:
        pass

    return mdot


def halo_infall_fakhouri(z, mhalo):

    mean_mdot = (
        46.1
        * (u.solMass / u.yr)
        * (mhalo / (1e12 * u.solMass)) ** 1.1
        * (1.0 + 1.11 * z)
        * ((Omegam0 * (1 + z) ** 3) + Omegade0) ** 0.5
    )
    mean_mdot = mean_mdot.to(u.solMass / u.Gyr)
    return mean_mdot


# plot halo infall as a function of redshift for a few halo masses
halo_masses = [1e10, 1e11, 1e12, 1e13]  # Msun
halo_masses = [m * u.Msun for m in halo_masses]

redshifts_new = []
redshifts_old = []
dot_mhalos_new = []
dot_mhalos_old = []
mhalos_new = []
mhalos_old = []




for mhalo in halo_masses:
    model_updated_infall = CGMRegulator(
        mhalo,
        (0.15, 13),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        updated_halo_infall=True,
        updated_2phase_CGM=False,
        updated_SF_law=False,
        updated_loadings=False,
        alpha_e=0.1,
        alpha_m=0.1,
    )
    run_updated_infall = model_updated_infall.run_halo()

    results_updated_infall = model_updated_infall.get_results()
    derived_updated_infall = model_updated_infall.get_derived_quantities()
    zs = results_updated_infall["z"]
    dot_m_halo = derived_updated_infall["dot_m_halo"]
    redshifts_new.append(zs)
    dot_mhalos_new.append(dot_m_halo)
    mhalos_new.append(results_updated_infall["m_halo"])

    model_old_infall = CGMRegulator(
        mhalo,
        (0.15, 13),
        KS_kappa_s=0.1,
        add_f_prevent_floor=1e-6,
        verbose=False,
        updated_halo_infall=False,
        updated_2phase_CGM=False,
        updated_SF_law=False,
        updated_loadings=False,
        alpha_e=0.1,
        alpha_m=0.1,
    )
    run_old_infall = model_old_infall.run_halo()
    results_old_infall = model_old_infall.get_results()
    derived_old_infall = model_old_infall.get_derived_quantities()
    redshifts_old.append(results_old_infall["z"])
    dot_mhalos_old.append(derived_old_infall["dot_m_halo"])
    mhalos_old.append(results_old_infall["m_halo"])

#%%  plot the halo growth differneces
fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=200) 
for i, mhalo in enumerate(halo_masses):

    zs = redshifts_new[i]
    dot_m_halo = dot_mhalos_new[i]
    zs_old = redshifts_old[i]
    dot_m_halo_old = dot_mhalos_old[i]

    ax[0].plot(zs, dot_m_halo, label="{:.0e} Msun".format(mhalo.value))
    # get color from the line just plotted
    line = ax[0].lines[-1]
    color = line.get_color()
    
    ax[0].plot(zs_old, dot_m_halo_old, ls="--", color=color)

    # plot the actual halo masses m_halo
    ax[1].plot(zs, mhalos_new[i])
    ax[1].plot(zs_old, mhalos_old[i], ls="--", color=color)
    

    
        
    # new v_circs in the third column
    mhalo_growth_new = mhalos_new[i] * u.Msun
    halo_rvir = virial_radius(zs, mhalo_growth_new).to(u.kpc)
    halo_vcirc = circular_velocity(mhalo_growth_new, halo_rvir).to(u.km / u.s)
    mhalo_growth_old = mhalos_old[i] * u.Msun
    halo_rvir_old = virial_radius(zs_old, mhalo_growth_old).to(u.kpc)
    halo_vcirc_old = circular_velocity(mhalo_growth_old, halo_rvir_old).to(u.km / u.s)
    
    ax[2].plot(zs, halo_vcirc, label="{:.0e} Msun".format(mhalo.value))
    ax[2].plot(zs_old, halo_vcirc_old, ls="--", color=color)
    

# build handles for each halo mass (solid = updated infall, dashed = old infall)
mass_handles = []
mass_labels = []
for i, m in enumerate(halo_masses):
    # solid line is at 2*i, dashed at 2*i+1
    color = ax[0].lines[2 * i].get_color()
    mass_handles.append(Line2D([0], [0], color=color, lw=2))
    mass_labels.append(f"{m.value:.0e} Msun")
    # handles to indicate the line style meaning
    style_handles = [
        Line2D([0], [0], color="k", lw=2, linestyle="-"),
        Line2D([0], [0], color="k", lw=2, linestyle="--"),
    ]
    style_labels = ["updated infall (Fakhouri)", "old infall (Dekel)"]

    # combine mass (color) handles with style (linestyle) handles into a single legend
    combined_handles = mass_handles + style_handles
    combined_labels = mass_labels + style_labels

ax[0].legend(
    combined_handles,
    combined_labels,
    loc="upper left",
    fontsize=8,
    ncols=4,
    bbox_to_anchor=(0, 1.2)
)
ax[0].set(
    xlabel="z",
    ylabel="halo growth rate [M_sun / Gyr]",
    yscale="log",
    xscale="log",
    xlim=(zs.min(), zs.max()),
)
ax[1].set(
    xlabel="z",
    ylabel="halo mass [M_sun]",
    yscale="log",
    xscale="log",
    xlim=(zs.min(), zs.max()),
)

ax[2].set(
    xlabel="z",
    ylabel=r"halo circular velocity $V_{\rm circ}$ [km/s]",
    yscale="log",
    xscale="log",
    xlim=(zs.min(), zs.max()),
)
ax[2].invert_xaxis()
# flip x axis
ax[0].invert_xaxis()
ax[1].invert_xaxis()
plt.show()
#%% now, plot how this impacts the mass and energy loadings
aM = 0.1
aE = 0.1

# to see how the mass and energy loadings change in each approach, we use the halo mass found through each accretion model, which we look at in greater detail below

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
for i, mhalo in enumerate(halo_masses):

    zs = redshifts_new[i]
    dot_m_halo = dot_mhalos_new[i]
    zs_old = redshifts_old[i]
    dot_m_halo_old = dot_mhalos_old[i]

    mhalo_growth_new = mhalos_new[i] * u.Msun
    mhalo_growth_old = mhalos_old[i] * u.Msun

    # new parametrizations
    halo_rvir = virial_radius(zs, mhalo_growth_new).to(u.kpc)
    halo_vcirc = circular_velocity(mhalo_growth_new, halo_rvir).to(u.km / u.s)
    halo_vir_temp = virial_T(mhalo_growth_new, halo_rvir).to(u.K)
    eta_m = vcirc_mass_loading(halo_vcirc, alpha_m=aM)
    eta_e = vcirc_energy_loading(halo_vcirc, alpha_e=aE)
    
    
    # ax[0].plot(zs, eta_m, label="{:.0e} Msun".format(mhalo.value))
    # # get color from the line just plotted
    # line = ax[0].lines[-1]
    # color = line.get_color()    
    # ax[0].plot(zs_old, eta_m_old, ls="--", color=color)
    
    # we can also calculate the loading based on old halo growth
    halo_rvir_old = virial_radius(zs_old, mhalo_growth_old).to(u.kpc)
    halo_vcirc_old = circular_velocity(mhalo_growth_old, halo_rvir_old).to(u.km / u.s)
    halo_vir_temp_old = virial_T(mhalo_growth_old, halo_rvir_old).to(u.K)   
    eta_m_new_using_old = vcirc_mass_loading(halo_vcirc_old, alpha_m=aM)
    eta_e_new_using_old = vcirc_energy_loading(halo_vcirc_old, alpha_e=aE)

    ax[0].plot(zs_old, eta_m_new_using_old,  ls="-")
    color = ax[0].lines[-1].get_color()
    ax[1].plot(zs_old, eta_e_new_using_old, ls="-", color=color)
    

    # old parametrizations
    eta_m_old = custom_mass_loading(mhalo_growth_old, A=10, alpha=-0.7)
    eta_e_old = custom_energy_loading(mhalo_growth_old, A=0.1, alpha=-0.5)

    ax[0].plot(zs_old, eta_m_old, ls="--", color=color, label="{:.0e} Msun".format(mhalo.value))
    ax[1].plot(zs_old, eta_e_old, ls="--", color=color)
    

ax[0].set(
    xlabel="z",
    ylabel=r"mass loading factor $\eta_M$",
    yscale="log",
    xscale="log",
    xlim=(zs.min(), zs.max()),
)
ax[1].set(
    xlabel="z",
    ylabel=r"energy loading factor $\eta_E$",
    yscale="log",
    xscale="log",
    xlim=(zs.min(), zs.max()),
)

# reverse x axis
ax[0].invert_xaxis()
ax[1].invert_xaxis()

# build handles for each halo mass (solid = updated infall, dashed = old infall)
mass_handles = []
mass_labels = []
for i, m in enumerate(halo_masses):
    # solid line is at 2*i, dashed at 2*i+1
    color = ax[0].lines[2 * i].get_color()
    mass_handles.append(Line2D([0], [0], color=color, lw=2))
    mass_labels.append(f"{m.value:.0e} Msun")
    # handles to indicate the line style meaning
    style_handles = [
        Line2D([0], [0], color="k", lw=2, linestyle="-"),
        Line2D([0], [0], color="k", lw=2, linestyle="--"),
    ]
    style_labels = ["new loadings (using old infall)", "old loadings (using old infall)"]

    # combine mass (color) handles with style (linestyle) handles into a single legend
    combined_handles = mass_handles + style_handles
    combined_labels = mass_labels + style_labels

    ax[0].legend(
        combined_handles,
        combined_labels,
        loc="upper left",
        fontsize=8,
        ncols=4,
        bbox_to_anchor=(0, 1.2)
    )

plt.show()