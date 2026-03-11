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
import astropy.constants as consts
from run_grids_of_models import (
    run_baseline_model_redshift_grid,
    run_2phase_model_redshift_grid,
)
import h5py
import os

# %%
importlib.reload(cgm_sf_regulator)
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
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

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


def vcirc_energy_loading(halo_vcirc, alpha_e=0.1):
    eta_e = alpha_e * (halo_vcirc.value / 200) ** (-3 / 2)

    # if eta e > 1 set to 1, halo_vcirc can be float or array
    if np.any(eta_e > 0):
        eta_e = np.where(eta_e > 1, 1, eta_e)
    else:  # it's a float
        if eta_e > 1:
            eta_e = 1
    return eta_e


def virial_T(mhalo, Rvir):

    G = consts.G
    kb = consts.k_B
    mp = consts.m_p
    return (2 / 5) * ((G * mhalo * mp) / (Rvir * kb))


def vcirc_mass_loading(halo_vcirc, alpha_m=0.1):
    return alpha_m * (halo_vcirc.value / 200) ** (-3 / 2)


def virial_radius(z, mhalo, Delc=200):

    return (mhalo / (LCDM.critical_density(z) * (4 / 3) * np.pi * Delc)) ** (1 / 3)


def circular_velocity(mhalo, Rvir):
    """circular velocity of the halo"""
    G = consts.G
    return np.sqrt(G * mhalo / Rvir).to(u.km / u.s)


def metallicity_energy_loading(ism_metalllicity_z_sun, alpha_e=0.3, exp=-0.5):
    """energy loading factor as a function of ISM metallicity, power-law form"""
    eta_e = alpha_e * (ism_metalllicity_z_sun) ** exp
    print(eta_e)
    # if eta e > 1 set to 1, ism_metallicity can be float or array

    eta_e = np.where(eta_e > 1, 1, eta_e)

    return eta_e


def Zsun_to_twelve_log_oh(Z):
    zsun = 0.013
    twelve_log_oh_sun = 8.69
    twelve_log_oh = twelve_log_oh_sun + np.log10(Z / zsun)
    return twelve_log_oh


def twelve_log_oh_to_Zsun(twelve_log_oh):
    zsun = 0.013
    twelve_log_oh_sun = 8.69
    Z = zsun * 10 ** (twelve_log_oh - twelve_log_oh_sun)
    Z_sun = Z / zsun
    return Z_sun


# add this function near the top of your file (before the loop)
def plot_two_phase_eta_scan(
    results_2phase,
    derived_2phase,
    results_2Phase_long,
    derived_2Phase_long,
    alphaM,
    alphaE,
    eta_Z, # this is a constant value
    mhalo_z0,
    xlim_for_zoom=(0.166, 2),
    eta_E_scaling_with_Z=(0.3, -0.5),
    show=True,
):
    zsun = 0.013
    red1 = "tab:red"
    red2 = "tab:orange"
    blu1 = "dodgerblue"
    blu2 = "tab:green"

    # long-run mass/energy rates
    dot_m_cgm_cooling_long = (
        results_2Phase_long["m_cgm_hot"] / derived_2Phase_long["tcool_real"]
    )
    dot_m_cgm_in_long = derived_2Phase_long["dot_m_cgm_in"]
    dot_m_sne_wind_long = derived_2Phase_long["dot_m_ism_wind"]
    dot_m_cgm_ej_long = derived_2Phase_long["dot_m_cgm_out"]
    dot_cgm_falling_long = (
        results_2Phase_long["m_cgm_cold"] / derived_2Phase_long["t_dynamical"]
    )
    dot_m_sfr_long = derived_2Phase_long["dot_m_sfr"]

    dot_e_sne_wind_long = derived_2Phase_long["dot_e_ism_wind"]
    dot_e_cgm_acc_long = derived_2Phase_long["dot_e_cgm_in"]
    dot_e_cgm_cool_long = derived_2Phase_long["dot_e_cgm_cooling"]
    dot_e_cgm_ej_long = derived_2Phase_long["dot_e_cgm_out"]

    # short-run (zoom) mass/energy rates and derived quantities
    dot_m_cgm_cooling = results_2phase["m_cgm_hot"] / derived_2phase["tcool_real"]
    dot_m_cgm_in = derived_2phase["dot_m_cgm_in"]
    dot_m_sne_wind = derived_2phase["dot_m_ism_wind"]
    dot_m_cgm_ej = derived_2phase["dot_m_cgm_out"]
    dot_cgm_falling = results_2phase["m_cgm_cold"] / derived_2phase["t_dynamical"]
    dot_m_sfr = derived_2phase["dot_m_sfr"]
    f_prevent = derived_2phase["dot_e_cgm_cooling"] / derived_2phase["dot_e_cgm_out"]

    dot_e_sne_wind = derived_2phase["dot_e_ism_wind"]
    dot_e_cgm_acc = derived_2phase["dot_e_cgm_in"]
    dot_e_cgm_cool = derived_2phase["dot_e_cgm_cooling"]
    dot_e_cgm_ej = derived_2phase["dot_e_cgm_out"]
    dot_e_cgm_total = dot_e_cgm_acc + dot_e_sne_wind - dot_e_cgm_ej - dot_e_cgm_cool

    ism_metallicity = results_2Phase_long["m_metals_ism"] / results_2Phase_long["m_ism"]
    ism_metallicity_z_sun = ism_metallicity / zsun

    cgm_metallicity = results_2Phase_long["m_metals_cgm"] / results_2Phase_long["m_cgm"]
    cgm_metallicity_z_sun = cgm_metallicity / zsun

    # use the long results to compute feedback properties
    zs = results_2Phase_long["z"]
    times = results_2Phase_long["t"] * u.Gyr
    mhalo_growth_new = results_2Phase_long["m_halo"] * u.Msun

    halo_rvir = virial_radius(zs, mhalo_growth_new).to(u.kpc)
    halo_vcirc = circular_velocity(mhalo_growth_new, halo_rvir).to(u.km / u.s)
    halo_vir_temp = virial_T(mhalo_growth_new, halo_rvir).to(u.K)
    eta_m = vcirc_mass_loading(halo_vcirc, alpha_m=aM)
    eta_e = vcirc_energy_loading(halo_vcirc, alpha_e=aE)
    eta_e_metallicity = metallicity_energy_loading(
        ism_metallicity_z_sun,
        alpha_e=eta_E_scaling_with_Z[0],
        exp=eta_E_scaling_with_Z[1],
    )

    # create figure and main axes
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.23, wspace=0.2)

    # long-run insets at top of each panel
    ax0 = ax[0].inset_axes([0, 1.2, 0.45, 1.4])
    ax1 = ax[1].inset_axes([0.55, 2.44, 0.45, 1.4])

    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sfr_long,
        lw=2,
        label=r"$\dot{M}_{\rm \star}$",
        color="mediumorchid",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_ej_long,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, ej}$",
        color=blu2,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_cgm_falling_long,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, falling}$",
        color="y",
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_cooling_long,
        lw=2,
        label=r"$\dot{M}_{\rm CGM, cooling}$",
        color=blu1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_cgm_in_long,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm CGM, acc}$",
        color=red1,
    )
    ax0.plot(
        results_2Phase_long["z"],
        dot_m_sne_wind_long,
        ls="--",
        lw=2,
        label=r"$\dot{M}_{\rm SNe, wind}$",
        color=red2,
    )
    ax0.set(
        ylabel=r"mass rates $[{\rm M_{\odot}}\: {\rm Gyr^{-1}}]$",
        yscale="log",
        ylim=(1e7, None),
        xlabel=r"$z$",
        xlim=(0, 15),
    )
    ax0.invert_xaxis()
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.xaxis.set_tick_params(labeltop=True)
    ax0.xaxis.labelpad = 10

    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_ej_long,
        color=blu2,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, ej}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_cool_long,
        color=blu1,
        lw=2,
        label=r"$\dot{E}_{\rm CGM, cooling}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_cgm_acc_long,
        color=red1,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm CGM, acc}$",
    )
    ax1.plot(
        results_2Phase_long["z"],
        dot_e_sne_wind_long,
        color=red2,
        ls="--",
        lw=2,
        label=r"$\dot{E}_{\rm SNe, wind}$",
    )
    ax1.set(
        ylabel=r"energy rates $[{\rm erg\: Gyr^{-1}}]$",
        yscale="log",
        ylim=(5e54, None),
        xlabel=r"$z$",
        xlim=(0.0, 15),
    )
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_tick_params(labeltop=True)
    ax1.invert_xaxis()
    ax1.xaxis.labelpad = 10

    # main short/zoom panels
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
    )
    ax[0].legend(
        frameon=False,
        loc="upper center",
        fontsize=10.5,
        ncol=6,
        bbox_to_anchor=(0.5, 1.22),
    )

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

    # ensure lines sit behind axes labels
    for axes in ax:
        for line in axes.lines:
            line.set_zorder(1)

    # f_prevent inset
    axin = ax[0].inset_axes([1.1, 1.75, 1, 1])
    axin.plot(results_2phase["t"], f_prevent, color="k", lw=3)
    axin.set(
        ylabel=r"$f_{\rm prevent}$ unclipped",
        xlim=xlim_for_zoom,
        ylim=(5e-3, None),
        yscale="log",
    )
    axin.axhspan(1e-6, 1, color="k", alpha=0.1)

    # CGM temperature inset
    ax2 = ax[1].inset_axes([1.1, 1.85, 1, 1])
    ax2.plot(
        results_2phase["t"],
        derived_2phase["cgm_temp"],
        color="crimson",
        lw=3,
        label=r"$T_{\rm CGM}$",
    )
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

    # cooling lambda inset
    ax3 = ax[1].inset_axes([1.1, 0.75, 1, 1])
    ax3.plot(
        results_2phase["t"], derived_2phase["cooling_lambda"], color="darkcyan", lw=3
    )
    ax3.set(
        ylabel=r"$\Lambda$ [erg cm$^3$ s$^{-1}$]",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
        ylim=(1e-25, 1e-20),
    )

    # halo mass annotation
    ax[0].text(
        0.5,
        0.1,
        r"$M_{{\rm halo}}(z=0) = {:.1e} \: M_{{\odot}}$".format(mhalo_z0.value),
        transform=ax[0].transAxes,
        fontsize=12,
        ha="center",
        va="bottom",
        color="k",
    )

    ax[0].text(
        0.95,
        0.1,
        r"$\alpha_M={:.3f}, \alpha_E={:.3f}$".format(alphaM, alphaE),
        transform=ax[0].transAxes,
        fontsize=12,
        ha="right",
        va="bottom",
        color="k",
    )

    # if show:
    #     plt.show()
    # loading inset
    ax4 = ax[1].inset_axes([1.1, -0.5, 1, 1])
    ax4.plot(times, eta_m, color=blu2, lw=3, label=r"$\eta_M$")
    ax4.plot(times, eta_e, color=red2, lw=3, label=r"$\eta_E$", ls="--")
    ax4.plot(
        times,
        eta_e_metallicity,
        color="blue",
        lw=3,
        label=r"$\eta_E {{\rm (Z)}}, {{\rm norm}} = {:.2f}, {{\rm exp}} = {:.2f}$".format(
            eta_E_scaling_with_Z[0], eta_E_scaling_with_Z[1]
        ),
        ls=":",
    )
    ax4.set(
        ylabel=r"loading",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        yscale="log",
        xlim=xlim_for_zoom,
        ylim=(1e-2, 10),
    )
    ax4.legend(loc="lower left", fontsize=14, ncols=3)

    # CGM and ISM metallicity inset
    ax5 = ax[1].inset_axes([1.1, -1.6, 1, 1])
    ax5.plot(
        results_2Phase_long["t"],
        ism_metallicity_z_sun,
        color="darkorange",
        lw=3,
        label=r"$Z_{\rm ISM}$",
    )
    ax5.plot(
        results_2Phase_long["t"],
        cgm_metallicity_z_sun,
        color="purple",
        lw=3,
        ls="--",
        label=r"$Z_{\rm CGM}$",
    )
    ax5.legend(loc="best")
    ax5.set(
        ylabel=r"Metallicity [$Z_{\odot}$]",
        xlabel=r"$t_{\rm univ}$ [Gyr]",
        xlim=xlim_for_zoom,
        yscale="log",
    )
    
    # large text for all the relevant mass and energy loading terms in the run 
    text = "Mass loading: " + r"$\alpha_M = {:.3f}$".format(alphaM) + "\n" + "Metal Loading: " + r"$\eta_Z = {:.3f}$".format(eta_Z) + "\n" + r"Energy loading: $\alpha_E = {:.3f}$".format(alphaE) + "\n" + r"Metallicity energy loading: " + "\n" + r"norm = {:.2f}, exp = {:.2f}".format(eta_E_scaling_with_Z[0], eta_E_scaling_with_Z[1])
    ax5.text(
        0.05,
        - 0.5,
        text,
        transform=ax5.transAxes,
        fontsize=15,
        ha="left",
        va="top",
        color="k",
    )

    return fig, ax


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
zobs = [0.01]
mass_bins = 10
mhalos = np.geomspace(1e10, 1e13, mass_bins)
mhalos = np.broadcast_to(mhalos, (len(zobs), mhalos.size)) * u.Msun

# %% do a single run of the 2 phase model and breakdown the energy and mass contributions
## this is figure 2 of the first draft of the SF paper


# alpha_Ms = np.linspace(0.12, 10, 20) #np.linspace(0.1, 10, 5)
# alpha_Es = np.ones_like(alpha_Ms) * 0.1


# eta_E_scaling_with_Z = np.linspace(-0.5, -1, 10)
# alpha_Es_met_norm = np.geomspace(0.1, 5, 10)
n_scan = 16

  # np.linspace(-0.5, -1.5, 10)
# low specific energy to high specific energy
# alpha_Ms = [0.1]#np.geomspace(0.0001, 10, 20)[::-1]
# alpha_Es = [0.1]#np.ones_like(alpha_Ms) * 0.1
# alpha_Es_met_norm = [0.3]

alpha_Ms = np.ones(n_scan) * 0.1
alpha_Es = np.ones(n_scan) * 0.1
eta_Zs = np.ones(n_scan) * 0.5 #np.linspace(0.1,0.9,n_scan)

# vary the normalization
# alpha_Es_met_norm = np.geomspace(0.01, 1, n_scan)
# alpha_Es_met_exp = np.ones(n_scan) * -1.0

#  vary the exponnents
# alpha_Es_met_norm = np.ones(n_scan) * 0.3
# alpha_Es_met_exp = np.linspace(-0.1, -1.5, n_scan) 

alpha_Es_met_norm = [0.1]
alpha_Es_met_exp = [-1.5]

# vary both
# alpha_Es_met_norm = np.linspace(0.1, 1, n_scan)
# alpha_Es_met_exp = np.linspace(-0.5, -1.5, n_scan)


for i, (aM, aE) in enumerate(zip(alpha_Ms[:1], alpha_Es[:1])):
    eta_Z = eta_Zs[i]
    model_2phase = CGMRegulator(
        mhalo_z0,
        (0.2, xlim_for_zoom[1]),
        add_f_prevent_floor=1e-8,
        verbose=False,
        alpha_e=aE,
        alpha_m=aM,
        eta_z =eta_Z,
        eta_E_scaling_with_Z=(
            alpha_Es_met_norm[i],
            alpha_Es_met_exp[i],
        ),  
    )

    model_2Phase_long = CGMRegulator(
        mhalo_z0,
        (0.15, 13),
        add_f_prevent_floor=1e-8,
        verbose=False,
        alpha_e=aE,
        alpha_m=aM,
        eta_z =eta_Z,
        eta_E_scaling_with_Z=(
            alpha_Es_met_norm[i],
            alpha_Es_met_exp[i],
        ),  
    )

    run_2phase = model_2phase.run_halo()
    results_2phase = model_2phase.get_results()
    derived_2phase = model_2phase.get_derived_quantities()

    run_2Phase_long = model_2Phase_long.run_halo()
    results_2Phase_long = model_2Phase_long.get_results()
    derived_2Phase_long = model_2Phase_long.get_derived_quantities()

    fig, ax = plot_two_phase_eta_scan(
        results_2phase,
        derived_2phase,
        results_2Phase_long,
        derived_2Phase_long,
        alphaM=aM,
        alphaE=aE,
        mhalo_z0=mhalo_z0,
        xlim_for_zoom=xlim_for_zoom,
        eta_Z=eta_Z,
        eta_E_scaling_with_Z=(alpha_Es_met_norm[i], alpha_Es_met_exp[i]),
        show=False,
    )

    # do comparison to z=0, observables
    kappa_sfr = 0.02
    n_sfr = 1.8
    r_disk_sfr = 0.018

    param_txt_with_ZdepetaE = (
        f"KS1998_loadingVcirc_etaZ_scan_z0_{alpha_Es_met_norm[i]:.3f}".replace(".", "p")
        + f"_{alpha_Es_met_exp[i]:.3f}".replace(".", "p")
        + f"_{eta_Z:.3f}".replace(".", "p")
        + f"_{kappa_sfr:.3f}".replace(".", "p")
        + f"_n{n_sfr:.3f}".replace(".", "p")
        + f"_r{r_disk_sfr:.3f}".replace(".", "p")
    )
    # another set if not varying the metallicity dependence of the energy loading
    # param_txt = (
    #     f"KS1998_loadingVcirc_etaZ_scan_z0_{eta_Z:.3f}".replace(".", "p")
    #     + f"_{kappa_sfr:.3f}".replace(".", "p")
    #     + f"_n{n_sfr:.3f}".replace(".", "p")
    #     + f"_r{r_disk_sfr:.3f}".replace(".", "p")
    # )

    
    file = "./runs/smhm_2phase_redshift_scan_" f"{param_txt_with_ZdepetaE}.h5"

    if not os.path.exists(file):
        print("running 2-phase model grid...", file)
        redshift_variation, zsims = run_2phase_model_redshift_grid(
            observe_at=zobs,  # redshift we want to observe
            mhalos=mhalos,
            write_to_file=file,
            disk_scale_length=r_disk_sfr,
            KS_n=n_sfr,
            KS_kappa_s=kappa_sfr,
            eta_z=eta_Z,
            eta_E_scaling_with_Z=(alpha_Es_met_norm[i], alpha_Es_met_exp[i]),
        )
        with h5py.File(file, "r") as f:
            smhm_normalized = f["SMHM"][
                :
            ]  
            mhalo_obs = f["Mhalo_obs"][:]
            mism = f["MISM_obs"][:]
            mstar = f["Mstar_obs"][:]
            mmetals_ism = f["MMetals_ism_obs"][:]
            print(f.keys())
            print(f["redshifts"][:])
    else:
        print("file already exists")
        with h5py.File(file, "r") as f:
            smhm_normalized = f["SMHM"][
                :
            ]  
            print(f.keys())
            mhalo_obs = f["Mhalo_obs"][:]
            mmetals_ism = f["MMetals_ism_obs"][:]
            mism = f["MISM_obs"][:]
            mstar = f["Mstar_obs"][:]
           
            print(f["redshifts"][:])

    # plot the SMHM relation at z=0 and compare to Behroozi
    ax_smhm = ax[1].inset_axes([0.0, -1.25, 1, 1])
    ax_smhm.fill_between(
        10**loghm,
        smhm_behroozi * smhm_err_low,
        smhm_behroozi * smhm_err_up,
        facecolor="grey",
        alpha=0.3,
        zorder=0,
    )
    ax_smhm.scatter(
        mhalo_obs[0], smhm_normalized[0], color="crimson", lw=3, label=r"model z=0"
    )
    ax_smhm.set(
        xlabel=r"$M_{\rm halo}$ [$M_{\odot}$]",
        ylabel=r"$M_{\rm \star} / M_{\rm halo} \: (f_b^{-1})$",
        xscale="log",
        yscale="log",
        xlim=(1e10, 1e13),
        ylim=(2e-3, 0.2),
    )
    ax_smhm.legend(loc="upper left", fontsize=10)

    # plot ISM gas fractions
    ax_mism = ax[1].inset_axes([0.0, -2.5, 1, 1])
    ax_mism.scatter(
        mism[0], mism[0] / mstar[0], color="crimson", lw=3, label=r"model z=0"
    )

    log_m_star_thry = np.linspace(6, 12, 100)
    m_star_thry = 10**log_m_star_thry

    # Calette but double power law, still LTG
    C = 1.69
    a = 0.18
    b = 0.61
    log_Mtrunc = 9.2
    intrinsic_scatter_dbl_pl = 0.44
    y_dbl_power = C / (
        (m_star_thry / 10**log_Mtrunc) ** (a) + (m_star_thry / 10**log_Mtrunc) ** b
    )
    ax_mism.plot(
        m_star_thry,
        y_dbl_power,
        ls="-.",
        color="thistle",
        lw=2,
        label="Calette+18 LTG DPL",
        zorder=0,
    )
    y_dbl_power_upper = y_dbl_power * 10**intrinsic_scatter_dbl_pl
    y_dbl_power_lower = y_dbl_power / 10**intrinsic_scatter_dbl_pl
    ax_mism.fill_between(
        m_star_thry,
        y_dbl_power_lower,
        y_dbl_power_upper,
        facecolor="thistle",
        alpha=0.5,
        zorder=0,
    )
    ax_mism.set(
        xlabel=r"$M_{\rm star}$ [$M_{\odot}$]",
        ylabel=r"$M_{\rm ISM} / M_{\rm star} \: (f_b^{-1})$",
        xscale="log",
        yscale="log",
        xlim=(1e7, 5e11),
        ylim=(1e-1, 8),
    )
    ax_mism.legend(loc="lower left", fontsize=10)

    
    # add ism metallicity vs mstar
    zsun = 0.013
    ax_zism = ax[1].inset_axes([0.0, -3.75, 1, 1])
    ism_metallicity_z_sun = (mmetals_ism[0] / mism[0]) / zsun
    twelve_log_oh = 8.69 + np.log10(ism_metallicity_z_sun)
    
    ax_zism.scatter(
        np.log10(mstar[0]), twelve_log_oh, color="crimson", lw=3, label=r"model z=0"   
    )
    # add  observations
    curti_2020_metallicity = np.loadtxt("./data/curti+20_fig3.csv", delimiter=",")
    curti_mstar =  curti_2020_metallicity[:, 0]
    curti_12_log_oh = curti_2020_metallicity[:, 1]
    curti_lower_values = curti_2020_metallicity[:, 2]
    curti_upper_values = curti_2020_metallicity[:, 3]
    ax_zism.plot(
        curti_mstar,
        curti_12_log_oh,
        color="k",
        label="Curti+20, z~0",
        lw=2,
        alpha=1,
    )
    ax_zism.set(
    # xscale="log",
    # yscale="log",
    xlabel=r"$M_\star$ [M$_\odot$]",
    # ylabel=r"ISM Metallicity [$Z_\odot$]",
    ylabel=r"ISM Metallicity 12 + log(O/H) ",
    xlim=(7.5,12),
    ylim = (7.6, 9.3)
    )
    ax_zism.fill_between(
    curti_mstar,
    curti_lower_values,
    curti_upper_values,
    color="k",
    alpha=0.2,
    label="Curti+20 scatter",
)
    ax_zism.legend(loc="lower right", fontsize=10)

    # plt.savefig(
    #     "./figures/scan_etaEofZ_exp_etaZ0p5/{:04d}_etaM_{:.5f}-etaE_{:.5f}-etaZ_{:.5f}.png".format(
    #         i, aM, aE, eta_Z
    #     ),
    #     dpi=200,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )

    plt.show()

# %%
